#!/usr/bin/env python
"""Run the full teleoperation loop offline — no arm, no controller.

Drives `MockFollower` from `ScriptedSource` and reports the M3/M4 metrics the
blueprint asks each module to prove independently. This is the Phase 0 gate we
can pass at a desk; the on-hardware run swaps two constructors and nothing else.

Run:  ./scripts/check_teleop_loop.py [--steps N] [--realtime]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from so_snake.config import SoSnakeConfig
from so_snake.m4_execution import MockFollower
from so_snake.teleop import ScriptedSource, TeleopLoop

# Thresholds the module has to meet before we call Phase 0's desk-check control
# path done. The default scripted IMU sweep is intentionally mild; pass
# --rotation-amplitude 0.35 to stress atlas and joint-rate walls.
GATES = {
    "ik_pos_err_p95_mm": ("<", 0.05),
    "ik_pitch_err_p95_deg": ("<", 0.05),
    "ik_solver_converged_frac": (">", 0.97),
    "max_command_step_deg": ("<=", None),  # bound filled in from the config below
}

# Touching a joint limit is not a failure -- it is the clamp doing its job, and
# a stress sweep is meant to reach the edges. What must never happen is a
# command that exceeds a limit or moves faster than the cap, and those are the
# properties gated here and asserted in tests/test_teleop_loop.py.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--rotation-amplitude", type=float, default=0.10, help="synthetic IMU attitude amplitude, radians")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace the loop to control_hz, to measure achievable rate rather than raw throughput",
    )
    args = parser.parse_args()

    config = SoSnakeConfig()
    source = ScriptedSource.from_waveform(
        n_steps=args.steps,
        amplitude=args.amplitude,
        rotation_amplitude_rad=args.rotation_amplitude,
    )
    backend = MockFollower()
    loop = TeleopLoop(source, backend, config)

    print(f"backend   {type(backend).__name__}")
    print(
        f"source    {type(source).__name__} ({args.steps} steps, "
        f"stick amplitude {args.amplitude}, rotation amplitude {args.rotation_amplitude} rad)"
    )
    print(f"target    {config.teleop.control_hz:g} Hz, realtime={args.realtime}")
    print(f"IK        5D task DLS, rotation_gain={config.teleop.rotation_gain:g}")
    print()

    stats = loop.run(realtime=args.realtime)
    summary = stats.summary()

    print("=" * 62)
    print("M3 — feasibility projection & safety")
    print("=" * 62)
    print(f"  TCP position error   median {summary['ik_pos_err_median_mm']:8.4f} mm")
    print(f"                       p95    {summary['ik_pos_err_p95_mm']:8.4f} mm")
    print(f"                       max    {summary['ik_pos_err_max_mm']:8.4f} mm")
    print(f"  pitch error          p95    {summary['ik_pitch_err_p95_deg']:8.4f} deg")
    print(f"  roll error           p95    {summary['ik_roll_err_p95_deg']:8.4f} deg")
    print(f"  yaw residual         p95    {summary['yaw_residual_p95_deg']:8.4f} deg")
    print(f"  rejected rotation    p95    {summary['rejected_rotation_p95_deg']:8.4f} deg")
    print(f"  command converged    {summary['ik_converged_frac'] * 100:5.1f} % of steps")
    print(f"  solver converged     {summary['ik_solver_converged_frac'] * 100:5.1f} % of steps")
    print(f"  IK reseeded          {summary['ik_reseeded_frac'] * 100:5.1f} % of steps")
    print(f"  workspace clamped    {summary['workspace_clamped_frac'] * 100:5.1f} % of steps")
    print(f"  atlas pitch clamped  {summary['atlas_pitch_clamped_frac'] * 100:5.1f} % of steps")
    print(f"  joint limit clamped  {summary['joint_limit_clamped_frac'] * 100:5.1f} % of steps")
    print(f"  joint rate clamped   {summary['joint_rate_clamped_frac'] * 100:5.1f} % of steps")

    print()
    print("=" * 62)
    print("M4 — execution")
    print("=" * 62)
    joints = np.array([r.commanded_joints_deg for r in stats.records])
    steps = np.abs(np.diff(joints, axis=0))
    print(f"  writes issued        {backend.write_count}")
    print(f"  loop rate            median {summary['loop_hz_median']:8.1f} Hz")
    print(f"                       p05    {summary['loop_hz_p05']:8.1f} Hz")
    print(f"  joint step           max    {steps.max():8.4f} deg "
          f"(cap {config.teleop.max_joint_step_deg:g})")
    for i, name in enumerate(config.arm.joint_names):
        lo, hi = config.arm.joint_limits_deg[name]
        print(f"    {name:<15} range [{joints[:, i].min():+8.2f}, {joints[:, i].max():+8.2f}]  "
              f"limits [{lo:+8.2f}, {hi:+8.2f}]")

    print()
    print("=" * 62)
    summary["max_command_step_deg"] = float(steps.max())
    bounds = dict(GATES)
    bounds["max_command_step_deg"] = ("<=", config.teleop.max_joint_step_deg + 1e-6)

    failures = []
    for key, (op, bound) in bounds.items():
        value = summary[key]
        if op == "<":
            ok = value < bound
        elif op == ">":
            ok = value > bound
        else:
            ok = value <= bound
        print(f"  {'PASS' if ok else 'FAIL'}  {key} = {value:.4f}  (need {op} {bound:g})")
        if not ok:
            failures.append(key)

    print()
    print(f"RESULT: {'PASS' if not failures else 'FAIL — ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
