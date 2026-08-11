#!/usr/bin/env python
"""Teleoperate the REAL SO-100 arm from a Nintendo Pro controller.

This is the first script that commands motion on the physical arm. It wires the
same pipeline the sim scripts use -- NintendoProSource -> TeleopLoop (5D IK +
workspace/atlas safety) -> SOFollowerBackend -- onto real hardware, through the
lerobot<->URDF joint map so the loop's URDF-frame kinematics drive the arm
correctly.

Safety is deliberate and layered:
  * the joint map is required (raw frame would drive the arm wrong);
  * `--max-relative-target` (default 5 deg) clamps every servo step in hardware,
    behind the loop's own 6 deg/step cap;
  * motion happens only while the controller clutch (ZL) is held; released, the
    target freezes;
  * on exit torque is disabled (the arm goes limp).

Read `scripts/preflight_real_arm.py` output first, and keep a hand on the power.

The serial port is auto-detected (`--port` overrides it, as does SO_SNAKE_ARM_PORT):

    PYTHONPATH=src .venv/bin/python scripts/teleop_real_arm.py \
        --max-relative-target 5 --steps 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from so_snake.config import ARM_JOINTS, GRIPPER_JOINT, JOINT_LIMITS_DEG, REPO_ROOT, SoSnakeConfig
from so_snake.devices import DeviceDetectionError, detect_arm_port
from so_snake.m4_execution import JointFrameMap, SOFollowerBackend, move_to_joints
from so_snake.teleop import NintendoProSource, TeleopLoop

DEFAULT_MAP = REPO_ROOT / "assets" / "so100_joint_map.json"
DEFAULT_START = REPO_ROOT / "assets" / "so100_start_pose.json"


def _move_to_start(backend: SOFollowerBackend, start_path: Path, config: SoSnakeConfig) -> None:
    """Walk the arm to the recorded start pose so teleop begins inside the workspace."""
    if not start_path.is_file():
        print(f"no start pose at {start_path}; skipping move-to-start (starting from current pose).")
        return
    data = json.loads(start_path.read_text())["joints_urdf_deg"]
    start = np.array([data[n] for n in (*ARM_JOINTS, GRIPPER_JOINT)], dtype=float)
    print(f"moving to start pose {np.round(start, 1).tolist()} ...")
    last = [0.0]

    def progress(remaining: float) -> None:
        if remaining <= last[0] - 5.0 or last[0] == 0.0:
            print(f"    max remaining {remaining:6.1f} deg", flush=True)
            last[0] = remaining

    # step_deg leads the servo enough to drive against gravity; keep it within the
    # backend's max_relative_target so the hardware clamp does not starve it.
    reached = move_to_joints(backend, start, step_deg=6.0, tol_deg=1.0, hz=config.teleop.control_hz, on_progress=progress)
    print("at start pose." if reached else "WARN: move-to-start did not fully converge.")


def _startup_settle(backend: SOFollowerBackend, config: SoSnakeConfig, *, settle_steps: int, max_out_of_range_deg: float) -> None:
    """Energize gently and refuse to start if anything looks wrong.

    Reads the arm, checks the joint map round-trips exactly here (a corrupt map
    would otherwise silently mis-drive), checks the pose is within (or barely
    outside) the URDF limits, then commands the arm to HOLD exactly where it is
    for a moment. With max_relative_target clamping every step, this turns the
    torque-on transient into a no-op instead of a lurch -- and any real problem
    aborts before the arm has moved.
    """
    measured = backend.read_joints_deg()  # URDF degrees
    jm = backend.joint_map
    if jm is not None:
        for name, v in zip(backend.joint_names, measured):
            back = jm.lerobot_to_urdf(name, jm.urdf_to_lerobot(name, v))
            if abs(back - float(v)) > 1e-6:
                raise RuntimeError(f"joint map is not bijective at {name} ({v:.3f} -> {back:.3f}); aborting")

    lo = np.array([JOINT_LIMITS_DEG[j][0] for j in ARM_JOINTS])
    hi = np.array([JOINT_LIMITS_DEG[j][1] for j in ARM_JOINTS])
    arm = np.asarray(measured, float)[: len(ARM_JOINTS)]
    out = np.maximum(lo - arm, arm - hi)
    print("measured pose (URDF deg):", np.round(arm, 1).tolist())
    worst = float(out.max())
    if worst > 0:
        which = [ARM_JOINTS[i] for i in range(len(ARM_JOINTS)) if out[i] > 0]
        print(f"  note: {which} just outside URDF limits by up to {worst:.1f} deg")
        if worst > max_out_of_range_deg:
            raise RuntimeError(
                f"a measured joint is {worst:.1f} deg outside URDF limits (> {max_out_of_range_deg}); "
                "check the joint map / calibration before energizing. Aborting."
            )

    for _ in range(settle_steps):
        backend.write_joints_deg(measured)  # goal == present -> no motion
        time.sleep(1.0 / config.teleop.control_hz)
    print(f"settled: held current pose for {settle_steps} steps. Teleop starting; hold ZL to move.")


def _confirm(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _print_summary(loop: TeleopLoop, backend: SOFollowerBackend) -> None:
    records = loop.stats.records
    print("\n" + "=" * 60)
    print("Real-arm teleop summary")
    print("=" * 60)
    print(f"  steps            {len(records)}")
    if not records:
        return
    s = loop.stats.summary()
    clutch = np.array([r.clutch_engaged for r in records], dtype=bool)
    print(f"  clutch engaged   {100.0 * clutch.mean():5.1f} %")
    print(f"  loop hz median   {s['loop_hz_median']:7.1f}")
    print(f"  IK pos err p95   {s['ik_pos_err_p95_mm']:7.3f} mm")
    print(f"  workspace clamp  {100.0 * s['workspace_clamped_frac']:5.1f} %")
    print(f"  atlas clamp      {100.0 * s['atlas_pitch_clamped_frac']:5.1f} %")
    print(f"  joint-rate clamp {100.0 * s['joint_rate_clamped_frac']:5.1f} %")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="",
                        help="arm serial port; auto-detected when omitted "
                             "(e.g. /dev/cu.usbmodem58760434321)")
    parser.add_argument("--id", default="so_snake", help="lerobot robot id (calibration)")
    parser.add_argument("--map", default=str(DEFAULT_MAP), help="joint-frame map JSON from map_joint_frames.py")
    parser.add_argument("--device-id", type=int, default=None, help="optional NintendoTeleop device id")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=5.0,
        help="hardware per-step clamp for the arm joints, degrees (lower = safer/slower first runs)",
    )
    parser.add_argument(
        "--gripper-speed-mult",
        type=float,
        default=3.0,
        help="gripper's per-step clamp as a multiple of --max-relative-target (faster open/close)",
    )
    parser.add_argument("--steps", type=int, default=600, help="stop after this many control frames")
    parser.add_argument("--start", default=str(DEFAULT_START), help="start-pose JSON to move to before teleop")
    parser.add_argument("--no-move-to-start", action="store_true", help="skip moving to the start pose; teleop from current pose")
    parser.add_argument("--settle-steps", type=int, default=15, help="hold current pose for this many steps before teleop")
    parser.add_argument("--max-startup-out-of-range", type=float, default=15.0,
                        help="abort if a measured joint is more than this many deg outside URDF limits")
    parser.add_argument("--no-realtime", action="store_true", help="do not sleep to the control rate")
    parser.add_argument("--yes", action="store_true", help="skip the safety confirmation prompt")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.max_relative_target <= 0:
        raise SystemExit("--max-relative-target must be positive")
    map_path = Path(args.map)
    if not map_path.is_file():
        raise SystemExit(
            f"joint map not found: {map_path}\n"
            "build it first: scripts/map_joint_frames.py draft / signs / check"
        )

    if args.gripper_speed_mult <= 0:
        raise SystemExit("--gripper-speed-mult must be positive")

    config = SoSnakeConfig()
    # Per-motor clamp: keep the arm joints at the safe value, let the gripper move
    # faster (it is absolute and not rate-limited by the loop, so this is its only
    # speed knob).
    max_relative_target = {j: args.max_relative_target for j in ARM_JOINTS}
    max_relative_target[GRIPPER_JOINT] = args.max_relative_target * args.gripper_speed_mult
    try:
        port = detect_arm_port(args.port)
    except DeviceDetectionError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    try:
        joint_map = JointFrameMap.load(map_path)
        source = NintendoProSource(controller="pro", device_id=args.device_id)
        backend = SOFollowerBackend(
            port=port,
            arm=config.arm,
            robot_id=args.id,
            max_relative_target=max_relative_target,
            joint_map=joint_map,
        )
        loop = TeleopLoop(source, backend, config)
    except Exception as exc:  # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("needs the .[teleop] extra (lerobot + hidapi + feetech-servo-sdk) and the arm plugged in.", file=sys.stderr)
        return 2

    print("=" * 60)
    print("REAL ARM TELEOP — the arm WILL move and torque WILL engage.")
    print("=" * 60)
    print(f"  port                 {port}{'' if args.port else '  (auto-detected)'}")
    print(f"  joint map            {map_path}")
    print(f"  max_relative_target  arm {args.max_relative_target:g} / gripper "
          f"{args.max_relative_target * args.gripper_speed_mult:g} deg/step (hardware clamp)")
    print(f"  max_joint_step       {config.teleop.max_joint_step_deg:g} deg/step (loop clamp)")
    print(f"  steps                {args.steps}   control {config.teleop.control_hz:g} Hz")
    move_desc = "from current pose (--no-move-to-start)" if args.no_move_to_start else f"move to start first ({args.start})"
    print(f"  startup              {move_desc}")
    print("\n  - Clear the workspace. Keep a hand on the power / e-stop.")
    print("  - Motion happens only while you hold the clutch (ZL). Released = frozen.")
    print("  - Start with tiny stick inputs and watch that it moves the way you expect.")
    if not args.yes and not _confirm("\nType 'yes' to connect (this energizes the arm): "):
        print("aborted; nothing energized.")
        return 1

    rc = 0
    try:
        backend.connect()  # energizes; torque holds present position
        source.connect()
        if not args.no_move_to_start:
            _move_to_start(backend, Path(args.start), config)
        _startup_settle(
            backend,
            config,
            settle_steps=args.settle_steps,
            max_out_of_range_deg=args.max_startup_out_of_range,
        )
        loop.run(max_steps=args.steps, realtime=not args.no_realtime)
    except KeyboardInterrupt:
        print("\ninterrupted.")
    except Exception as exc:  # noqa: BLE001
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        rc = 2
    finally:
        # disconnect disables torque (disable_torque_on_disconnect=True) -> arm goes limp.
        for closer in (source.disconnect, backend.disconnect):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass

    _print_summary(loop, backend)
    return rc


if __name__ == "__main__":
    sys.exit(main())
