#!/usr/bin/env python
"""Replay a recorded episode onto an arm — mock, MuJoCo, or the real SO-100.

Two modes, and they answer different questions:

  --mode joint   send the recorded joint commands back out. Does the arm
                 reproduce what it did? A divergence is the hardware's.
  --mode task    re-solve the recorded 5D task targets through today's IK and
                 feasibility atlas. Would today's controller have done the same?
                 This is the regression test for a solver or projector change.

    PYTHONPATH=src python scripts/replay_episode.py --list
    PYTHONPATH=src python scripts/replay_episode.py --id ep_20260727_143000 --check
    PYTHONPATH=src python scripts/replay_episode.py --id ep_... --backend mujoco
    PYTHONPATH=src python scripts/replay_episode.py --id ep_... --backend real \\
        --speed 0.5

`--check` runs the static inspection and moves nothing. On the real arm, run it
first: it reports joint-order mismatches, commands recorded outside the current
limits, and whether the requested speed will exceed the joint rate cap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from so_snake.config import SoSnakeConfig
from so_snake.data import (
    DEFAULT_EPISODE_ROOT,
    EpisodeReplayer,
    EpisodeStore,
    ReplayConfig,
    inspect_episode,
)
from so_snake.devices import DeviceDetectionError, detect_arm_port
from so_snake.rig import DEFAULT_JOINT_MAP, RigSpec, build_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--list", action="store_true", help="list the library and exit")
    parser.add_argument("--id", default="", help="episode id; defaults to the newest")
    parser.add_argument("--backend", choices=("mock", "mujoco", "real"), default="mock")
    parser.add_argument("--mode", choices=("joint", "task"), default="joint")
    parser.add_argument("--speed", type=float, default=1.0, help="playback rate multiplier")
    parser.add_argument("--check", action="store_true", help="inspect only; move nothing")
    parser.add_argument("--no-realtime", action="store_true", help="play as fast as possible")
    parser.add_argument("--no-clearance-check", action="store_true",
                        help="skip the MuJoCo mesh clearance guard")

    real = parser.add_argument_group("real arm")
    real.add_argument("--port", default="", help="serial port; auto-detected when omitted")
    real.add_argument("--id-robot", dest="robot_id", default="so_snake")
    real.add_argument("--map", type=Path, default=DEFAULT_JOINT_MAP)
    real.add_argument("--max-relative-target", type=float, default=5.0)
    real.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SoSnakeConfig()
    store = EpisodeStore(args.root)

    library = store.list_meta()
    if args.list or not args.id:
        if not library:
            print(f"no episodes under {args.root}")
            return 1 if args.list else 2
        if args.list:
            print(f"{'id':<24} {'steps':>6} {'dur':>7}  {'backend':<8} {'task'}")
            for meta in library:
                print(
                    f"{meta.id:<24} {meta.n_steps:>6} {meta.duration_s:>6.1f}s  "
                    f"{meta.backend:<8} {meta.task or meta.name}"
                )
            return 0

    episode_id = args.id or library[0].id
    try:
        episode = store.load(episode_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    replay = ReplayConfig(
        mode=args.mode,
        speed=args.speed,
        realtime=not args.no_realtime,
        check_clearance=not args.no_clearance_check,
    )

    print(f"episode   {episode.meta.id}  ({episode.meta.n_steps} steps, "
          f"{episode.meta.duration_s:.1f} s, recorded on {episode.meta.backend})")
    if episode.meta.task:
        print(f"task      {episode.meta.task}")
    print(f"replay    {args.mode} mode at {args.speed:g}x onto {args.backend}")

    try:
        # Detected here rather than inside `build_backend` so the port is known
        # in time to be printed above the confirmation prompt, and so a machine
        # with no arm attached says so before the episode is loaded onto it.
        port = detect_arm_port(args.port) if args.backend == "real" else args.port
    except DeviceDetectionError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    spec = RigSpec(
        backend=args.backend,
        port=port,
        robot_id=args.robot_id,
        joint_map_path=args.map,
        max_relative_target_deg=args.max_relative_target,
    )

    backend = None
    if not args.check:
        try:
            backend = build_backend(spec, config)
        except Exception as exc:  # noqa: BLE001
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    issues = inspect_episode(
        episode,
        config,
        backend_joint_names=backend.joint_names if backend is not None else None,
        target_physical=spec.is_physical,
        replay=replay,
    )
    print()
    if not issues:
        print("inspection: clean")
    for issue in issues:
        print(f"  {issue.level.upper():<8} {issue.message}")
    if any(issue.level == "error" for issue in issues):
        print("\nrefusing to replay: fix the errors above.", file=sys.stderr)
        return 2
    if args.check:
        return 0

    if spec.is_physical:
        print()
        print("=" * 60)
        print("REAL ARM — it WILL move, along the whole recorded trajectory.")
        print("=" * 60)
        print(f"  port  {spec.port}{'' if args.port else '  (auto-detected)'}")
        print("  The arm first walks to the episode's first pose, then plays it back.")
        print("  Clear the workspace and keep a hand on the power.")
        if not args.yes and input("\nType 'yes' to connect: ").strip().lower() not in ("y", "yes"):
            print("aborted; nothing energized.")
            return 1

    replayer = EpisodeReplayer(episode, backend, config, replay)
    last_progress = [0.0]

    def progress(phase: str, value: float) -> None:
        if phase == "approach" and (last_progress[0] == 0.0 or value <= last_progress[0] - 5.0):
            print(f"  approach: max remaining {value:6.1f} deg", flush=True)
            last_progress[0] = value

    rc = 0
    print("\napproaching the first frame ...")
    try:
        report = replayer.run(on_progress=progress)
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            backend.disconnect()
        except Exception:  # noqa: BLE001
            pass

    print()
    print("=" * 60)
    if report.approach_note:
        # The replay ran, but not from exactly the first frame. Said out loud:
        # otherwise the first frames quietly close a gap nobody was told about.
        print(f"  approach  {report.approach_note}")
    print(f"replay {'completed' if report.completed else 'stopped'}: {report.n_steps} steps")
    if report.aborted_reason:
        print(f"  reason  {report.aborted_reason}")
        rc = 1
    print("=" * 60)
    for key, value in report.summary().items():
        print(f"  {key:<28} {value:.4f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
