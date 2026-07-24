#!/usr/bin/env python
"""Verify Nintendo Pro controller teleoperation against the MuJoCo simulation.

This is the hardware-input counterpart to `check_teleop_loop.py`: the source is
the real controller and the backend is still simulated, so no arm is required.

Run:
    PYTHONPATH=src scripts/check_pro_controller_sim.py --steps 300

Hold the controller's teleop enable/clutch while moving the sticks if you want
to see the simulated arm target move. A run with clutch released is still useful
for proving that USB controller reads, loop timing, IK, and simulation writes
are wired together.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from so_snake.config import SoSnakeConfig
from so_snake.m4_execution import MockFollower
from so_snake.sim import MujocoBackend
from so_snake.teleop import NintendoProSource, TeleopLoop


def _import_hint(exc: Exception) -> str:
    return (
        f"{type(exc).__name__}: {exc}\n\n"
        "This script needs the lab/lerobot environment for NintendoTeleop and "
        "`.[sim]` for MuJoCo. Typical invocations:\n"
        "  /home/hanyu/Codes/lerobot/.venv/bin/python scripts/check_pro_controller_sim.py --steps 300\n"
        "  or install this repo into that env with sim deps first."
    )


def _print_summary(loop: TeleopLoop, backend, *, max_samples: int = 5) -> list[str]:
    stats = loop.stats
    summary = stats.summary()
    failures: list[str] = []

    if not stats.records:
        print("=" * 66)
        print("Controller")
        print("=" * 66)
        print("  steps               0")
        return ["no controller samples were processed"]

    print("=" * 66)
    print("Controller")
    print("=" * 66)
    sticks = np.array([record.raw["action.raw.sticks"] for record in stats.records])
    clutch = np.array([record.clutch_engaged for record in stats.records], dtype=bool)
    gripper = np.array([record.gripper_cmd_deg for record in stats.records], dtype=float)
    print(f"  steps               {len(stats.records)}")
    print(f"  clutch engaged      {100.0 * clutch.mean():6.1f} %")
    print(f"  stick abs max       {np.abs(sticks).max(axis=0)}  order [lx ly rx rz]")
    print(f"  gripper cmd range   [{gripper.min():7.2f}, {gripper.max():7.2f}] deg")

    print()
    print("=" * 66)
    print("Loop")
    print("=" * 66)
    print(f"  writes issued       {backend.write_count}")
    print(f"  loop rate median    {summary['loop_hz_median']:8.1f} Hz")
    print(f"  loop rate p05       {summary['loop_hz_p05']:8.1f} Hz")
    print(f"  IK pos err p95      {summary['ik_pos_err_p95_mm']:8.4f} mm")
    print(f"  IK pitch err p95    {summary['ik_pitch_err_p95_deg']:8.4f} deg")
    print(f"  IK roll err p95     {summary['ik_roll_err_p95_deg']:8.4f} deg")
    print(f"  solver converged    {100.0 * summary['ik_solver_converged_frac']:6.1f} %")
    print(f"  workspace clamped   {100.0 * summary['workspace_clamped_frac']:6.1f} %")
    print(f"  atlas clamped       {100.0 * summary['atlas_pitch_clamped_frac']:6.1f} %")
    print(f"  joint rate clamped  {100.0 * summary['joint_rate_clamped_frac']:6.1f} %")

    commands = np.array([record.commanded_joints_deg for record in stats.records])
    if len(commands) > 1:
        max_step = float(np.abs(np.diff(commands, axis=0)).max())
    else:
        max_step = 0.0
    cap = loop.config.teleop.max_joint_step_deg
    if max_step > cap + 1e-6:
        failures.append(f"joint command step {max_step:.4f} deg exceeds cap {cap:g}")

    lo, hi = loop.config.arm.limits_deg_array()
    if np.any(commands < lo - 1e-9) or np.any(commands > hi + 1e-9):
        failures.append("joint command outside configured limits")

    print()
    print("=" * 66)
    print("Simulation")
    print("=" * 66)
    if hasattr(backend, "collisions"):
        collisions = backend.collisions
        print(f"  self-collisions     {len(collisions)}")
        for step, collision in collisions[:max_samples]:
            print(f"    step {step:04d}: {collision}")
        if collisions:
            failures.append(f"{len(collisions)} simulated self-collision steps")
    else:
        print("  self-collisions     n/a (mock backend)")

    if not np.isfinite(commands).all():
        failures.append("non-finite joint command")
    if backend.write_count != len(stats.records):
        failures.append(f"write count {backend.write_count} != records {len(stats.records)}")
    if len(stats.records) == 0:
        failures.append("no controller samples were processed")

    if not clutch.any():
        print()
        print("NOTE: clutch was never engaged, so target motion was intentionally held.")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300, help="number of controller frames to process")
    parser.add_argument("--device-id", type=int, default=None, help="optional lerobot NintendoTeleop device id")
    parser.add_argument("--backend", choices=("mujoco", "mock"), default="mujoco")
    parser.add_argument("--no-realtime", action="store_true", help="do not sleep to the configured control rate")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    config = SoSnakeConfig()
    try:
        source = NintendoProSource(controller="pro", device_id=args.device_id)
        backend = MujocoBackend(arm=config.arm) if args.backend == "mujoco" else MockFollower(arm=config.arm)
        loop = TeleopLoop(source, backend, config)
    except Exception as exc:
        print(_import_hint(exc), file=sys.stderr)
        return 2

    print(f"source     NintendoProSource(controller=pro, device_id={args.device_id})")
    print(f"backend    {type(backend).__name__}")
    print(f"target     {config.teleop.control_hz:g} Hz")
    print(f"steps      {args.steps}")
    print()

    try:
        loop.run(max_steps=args.steps, realtime=not args.no_realtime)
    except KeyboardInterrupt:
        print("\nInterrupted; reporting partial run.")
    except Exception as exc:
        print(_import_hint(exc), file=sys.stderr)
        return 2
    finally:
        try:
            source.disconnect()
        except Exception:
            pass
        try:
            backend.disconnect()
        except Exception:
            pass
        sim = getattr(backend, "sim", None)
        if sim is not None:
            sim.close()

    failures = _print_summary(loop, backend)
    print()
    print(f"RESULT: {'PASS' if not failures else 'FAIL - ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
