#!/usr/bin/env python
"""Record one teleoperation episode to disk.

The same pipeline the sim and hardware teleop scripts use, with an
`EpisodeRecorder` hung off the loop's step callback. Which arm and which input
device is one flag each:

    # offline: mock arm, scripted waveform -- no hardware, no simulator
    PYTHONPATH=src python scripts/record_episode.py --backend mock --steps 300

    # simulator with mesh clearance checking
    PYTHONPATH=src python scripts/record_episode.py --backend mujoco --source pro

    # the real arm. Read scripts/preflight_real_arm.py output first.
    PYTHONPATH=src python scripts/record_episode.py \\
        --backend real --source pro --port /dev/ttyACM0 --task "pick the red cube"

Episodes land in `data/episodes/<id>/`. `scripts/replay_episode.py` plays them
back; the GUI (`tools/gui`) does both with a button.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from so_snake.config import SoSnakeConfig
from so_snake.data import DEFAULT_EPISODE_ROOT, EpisodeRecorder
from so_snake.rig import DEFAULT_JOINT_MAP, RigSpec, build_backend, build_source
from so_snake.teleop import TeleopLoop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=("mock", "mujoco", "real"), default="mock")
    parser.add_argument("--source", choices=("scripted", "pro"), default="scripted")
    parser.add_argument("--steps", type=int, default=600, help="control frames to record")
    parser.add_argument("--name", default="", help="short label for the episode")
    parser.add_argument("--task", default="", help="what the demonstration shows")
    parser.add_argument("--notes", default="")
    parser.add_argument("--root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--no-realtime", action="store_true", help="do not pace to control_hz")

    real = parser.add_argument_group("real arm")
    real.add_argument("--port", default="", help="serial port, e.g. /dev/ttyACM0")
    real.add_argument("--id", dest="robot_id", default="so_snake", help="lerobot robot id")
    real.add_argument("--map", type=Path, default=DEFAULT_JOINT_MAP, help="joint-frame map JSON")
    real.add_argument("--max-relative-target", type=float, default=5.0,
                      help="hardware per-step clamp, degrees")
    real.add_argument("--yes", action="store_true", help="skip the confirmation prompt")

    scripted = parser.add_argument_group("scripted source")
    scripted.add_argument("--amplitude", type=float, default=0.2)
    scripted.add_argument("--rotation-amplitude", type=float, default=0.10)

    parser.add_argument("--device-id", type=int, default=None, help="NintendoTeleop device id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    config = SoSnakeConfig()
    spec = RigSpec(
        backend=args.backend,
        source=args.source,
        port=args.port,
        robot_id=args.robot_id,
        joint_map_path=args.map,
        max_relative_target_deg=args.max_relative_target,
        device_id=args.device_id,
        scripted_steps=max(args.steps, 2),
        scripted_amplitude=args.amplitude,
        scripted_rotation_amplitude_rad=args.rotation_amplitude,
        scripted_loop=False,
    )

    try:
        backend = build_backend(spec, config)
        source = build_source(spec, config)
    except Exception as exc:  # noqa: BLE001 - report the cause, do not traceback at a user
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"backend   {args.backend}")
    print(f"source    {args.source}")
    print(f"steps     {args.steps} at {config.teleop.control_hz:g} Hz")
    print(f"root      {args.root}")
    if spec.is_physical:
        print()
        print("=" * 60)
        print("REAL ARM — it WILL move and torque WILL engage.")
        print("=" * 60)
        print(f"  port                 {args.port}")
        print(f"  max_relative_target  {args.max_relative_target:g} deg/step (hardware clamp)")
        print("  Clear the workspace. Keep a hand on the power. Motion only while ZL is held.")
        print("  This script starts teleop from the CURRENT pose; for the guided")
        print("  move-to-start + settle sequence use scripts/teleop_real_arm.py.")
        if not args.yes and input("\nType 'yes' to connect: ").strip().lower() not in ("y", "yes"):
            print("aborted; nothing energized.")
            return 1

    recorder = EpisodeRecorder(
        args.root,
        config=config,
        backend=args.backend,
        source=args.source,
        simulated=not spec.is_physical,
        joint_names=backend.joint_names,
    )
    loop = TeleopLoop(source, backend, config)

    meta = recorder.start(name=args.name, task=args.task, notes=args.notes)
    print(f"\nrecording {meta.id} ... (Ctrl-C stops and keeps what was captured)")

    rc = 0
    try:
        loop.run(max_steps=args.steps, realtime=not args.no_realtime, on_step=recorder.append)
    except KeyboardInterrupt:
        print("\ninterrupted; keeping the partial episode.")
        recorder.abort("interrupted by the operator")
    except Exception as exc:  # noqa: BLE001
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        recorder.abort(f"{type(exc).__name__}: {exc}")
        rc = 2
    finally:
        for closer in (source.disconnect, backend.disconnect):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass

    written = recorder.stop(keep=True)
    if written is None:
        print("nothing recorded.")
        return rc or 1

    print(f"\nwrote {args.root / written.id}")
    print(f"  steps      {written.n_steps}")
    print(f"  duration   {written.duration_s:.2f} s")
    if written.aborted_reason:
        print(f"  aborted    {written.aborted_reason}")
    for key in ("ik_pos_err_p95_mm", "loop_hz_median", "workspace_clamped_frac",
                "joint_rate_clamped_frac"):
        if key in written.summary:
            print(f"  {key:<24} {written.summary[key]:.4f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
