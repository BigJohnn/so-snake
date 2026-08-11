#!/usr/bin/env python
"""Move the real SO-100 arm to the recorded start pose, in pure joint space.

A plain, IK-free joint move: each control step the arm is commanded a small step
from where it currently is toward the recorded start joints, so it walks there
smoothly and monotonically -- no task-space target, no IK branches, no chance of
a configuration flip. Rate-limited twice over (`--step-deg` in the loop and the
hardware `--max-relative-target`).

Record the start pose first. `--capture` reads the arm where it stands and
writes assets/so100_start_pose.json. It commands no motion: put the arm where
every take should begin (by hand, while it is limp), then run it. Connecting
does engage torque -- lerobot's `configure()` leaves it on -- so the arm holds
the pose you put it in for the second or so the read takes, and is explicitly
released again on the way out. The GUI has the same thing as a button, and there
it is usable mid-teleoperation.

    PYTHONPATH=src .venv/bin/python scripts/move_to_start.py --capture

Then, with the arm plugged in and the workspace clear (the serial port is
auto-detected; --port only to override it):
    PYTHONPATH=src .venv/bin/python scripts/move_to_start.py

The arm holds at the start pose (torque on) until you press ENTER, then torque is
released. Useful before teleop so the arm begins inside the workspace.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from so_snake.config import ARM_JOINTS, GRIPPER_JOINT, REPO_ROOT, SoSnakeConfig
from so_snake.devices import DeviceDetectionError, detect_arm_port
from so_snake.start_pose import StartPoseError, save_start_pose
from so_snake.m4_execution import JointFrameMap, SOFollowerBackend, move_to_joints

DEFAULT_MAP = REPO_ROOT / "assets" / "so100_joint_map.json"
DEFAULT_START = REPO_ROOT / "assets" / "so100_start_pose.json"
JOINT_ORDER = (*ARM_JOINTS, GRIPPER_JOINT)


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _capture(backend: SOFollowerBackend, start_path: Path, config: SoSnakeConfig, port: str) -> int:
    """Write the arm's current joints as the start pose. Commands no motion.

    Not quite "touches nothing": lerobot's `configure()` re-enables torque at
    the end of `connect()`, so the arm holds whatever pose it is in while this
    reads it. No goal position is ever sent, and `disconnect()` drops torque
    explicitly, so the arm is left limp rather than holding a pose nobody asked
    it to hold.
    """
    print(f"reading the arm on {port} (no motion commanded) ...")
    try:
        backend.connect()
        measured = backend.read_joints_deg()
    finally:
        try:
            backend.disconnect()
        except Exception:  # noqa: BLE001 - the read is what matters
            pass

    try:
        document = save_start_pose(
            measured, path=start_path, arm=config.arm, limits=config.limits, source="move_to_start --capture"
        )
    except StartPoseError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    print(f"wrote {start_path}")
    for name, value in document["joints_urdf_deg"].items():
        print(f"  {name:<15} {value:8.3f} deg")
    print("Homing (GUI) and move_to_start now go here.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="",
                        help="arm serial port; auto-detected when omitted "
                             "(e.g. /dev/cu.usbmodem58760434321)")
    parser.add_argument("--id", default="so_snake", help="lerobot robot id (calibration)")
    parser.add_argument("--map", default=str(DEFAULT_MAP), help="joint-frame map JSON")
    parser.add_argument("--start", default=str(DEFAULT_START), help="start-pose JSON")
    parser.add_argument("--capture", action="store_true",
                        help="record the arm's CURRENT joints as the start pose and exit "
                             "(read-only: no torque, no motion)")
    parser.add_argument("--max-relative-target", type=float, default=8.0, help="hardware per-step clamp, deg (>= --step-deg)")
    parser.add_argument("--step-deg", type=float, default=6.0,
                        help="goal lead per step toward the target, deg; also the servo drive (too small stalls under gravity)")
    parser.add_argument("--tol-deg", type=float, default=1.0, help="stop when every joint is within this, deg")
    parser.add_argument("--no-hold", action="store_true", help="release torque immediately instead of holding for ENTER")
    parser.add_argument("--yes", action="store_true", help="skip the safety confirmation prompt")
    args = parser.parse_args()

    for name, val in (("--step-deg", args.step_deg), ("--tol-deg", args.tol_deg),
                      ("--max-relative-target", args.max_relative_target)):
        if val <= 0:
            raise SystemExit(f"{name} must be positive")
    if args.max_relative_target < args.step_deg:
        print(f"note: --max-relative-target ({args.max_relative_target:g}) < --step-deg ({args.step_deg:g}); "
              "the hardware clamp will limit the servo drive to the smaller value and the arm may stall.",
              file=sys.stderr)
    map_path, start_path = Path(args.map), Path(args.start)
    if not map_path.is_file():
        raise SystemExit(f"joint map not found: {map_path}")
    if not args.capture and not start_path.is_file():
        raise SystemExit(f"start pose not found: {start_path}")

    config = SoSnakeConfig()
    hz = config.teleop.control_hz
    try:
        port = detect_arm_port(args.port)
    except DeviceDetectionError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    try:
        joint_map = JointFrameMap.load(map_path)
        backend = SOFollowerBackend(
            port=port, arm=config.arm, robot_id=args.id,
            max_relative_target=args.max_relative_target, joint_map=joint_map,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("needs the .[teleop] extra and the arm plugged in.", file=sys.stderr)
        return 2

    if args.capture:
        return _capture(backend, start_path, config, port)

    start_data = json.loads(start_path.read_text())["joints_urdf_deg"]
    start = np.array([start_data[n] for n in JOINT_ORDER], dtype=float)  # URDF degrees

    print("=" * 60)
    print("MOVE TO START — the arm WILL move and torque WILL engage.")
    print("=" * 60)
    print(f"  port: {port}{'' if args.port else '  (auto-detected)'}")
    print(f"  start (URDF deg): {np.round(start, 1).tolist()}")
    print(f"  step {args.step_deg:g} deg/loop, hardware clamp {args.max_relative_target:g} deg, {hz:g} Hz")
    print("  Clear the workspace. Keep a hand on the power / e-stop.")
    if not args.yes and not _confirm("\nType 'yes' to connect and move: "):
        print("aborted; nothing energized.")
        return 1

    rc = 0
    try:
        backend.connect()  # torque on, holds present position
        measured = backend.read_joints_deg()
        print(f"  from  (URDF deg): {np.round(measured, 1).tolist()}")
        print(f"  max joint distance: {float(np.abs(start - measured).max()):.1f} deg")

        next_print = [time.monotonic()]

        def progress(remaining: float) -> None:
            if time.monotonic() >= next_print[0]:
                print(f"    moving... max remaining {remaining:6.1f} deg", flush=True)
                next_print[0] = time.monotonic() + 0.5

        reached = move_to_joints(
            backend, start, step_deg=args.step_deg, tol_deg=args.tol_deg, hz=hz, on_progress=progress
        )
        if reached:
            print("reached start pose.")
        else:
            print("WARN: did not fully converge (obstruction, joint limit, or clamp too small).", file=sys.stderr)
            rc = 1

        backend.write_joints_deg(start)  # final hold command
        if not args.no_hold:
            try:
                input("holding at start (torque on). Press ENTER to release (torque off)...")
            except (EOFError, KeyboardInterrupt):
                pass
    except KeyboardInterrupt:
        print("\ninterrupted.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        rc = 2
    finally:
        try:
            backend.disconnect()  # disables torque -> arm goes limp
        except Exception:  # noqa: BLE001
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
