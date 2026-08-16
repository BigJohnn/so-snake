#!/usr/bin/env python
"""Replay an exported `LeRobotDataset` onto an arm — mock, MuJoCo, or the real SO-100.

This is the end of the export contract. `--verify` proves the rows on disk still
invert to the targets that were recorded; this proves an arm will actually
follow them, through the same safety layer a recorded take is replayed through.
Between them, "exported" and "replayable" mean the same thing.

    PYTHONPATH=src python scripts/replay_lerobot_dataset.py --dataset data/lerobot/x --list
    PYTHONPATH=src python scripts/replay_lerobot_dataset.py --dataset data/lerobot/x --check
    PYTHONPATH=src python scripts/replay_lerobot_dataset.py --dataset data/lerobot/x \\
        --episode 3 --backend mujoco
    PYTHONPATH=src python scripts/replay_lerobot_dataset.py --dataset data/lerobot/x \\
        --backend real --speed 0.5

Only task mode exists here, and that is not a limitation. The dataset carries no
joint stream on purpose -- the policy is trained in task space -- so the joints
are solved from the targets by today's IK, which is exactly what task-mode
replay of a recorded episode does. Replaying a joint stream this script had just
computed would be replaying its own arithmetic.

`--check` inspects and moves nothing. On the real arm, run it first.
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
    format_verify,
    inspect_episode,
    read_manifest,
    verify,
)
from so_snake.data.export import episode_from_dataset
from so_snake.devices import DeviceDetectionError, detect_arm_port
from so_snake.rig import DEFAULT_JOINT_MAP, RigSpec, build_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="exported dataset root (the directory holding export.json)")
    parser.add_argument("--list", action="store_true",
                        help="show what the dataset holds and exit")
    parser.add_argument("--episode", type=int, default=0,
                        help="dataset episode index to replay (default 0)")
    parser.add_argument("--verify", action="store_true",
                        help="run the full read-back check before replaying")
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT,
                        help="episode store the dataset was made from, for --verify")
    parser.add_argument("--backend", choices=("mock", "mujoco", "real"), default="mock")
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

    try:
        manifest = read_manifest(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    if args.list:
        print(f"dataset   {manifest['repo_id']}  at {args.dataset}")
        print(f"  {manifest['n_episodes']} episodes / {manifest['n_frames']} frames "
              f"at {manifest['fps']} Hz")
        print(f"  action space  {manifest['action_space']}")
        print(f"  cameras       {', '.join(manifest.get('cameras', ()))}")
        print(f"  task          {manifest.get('task') or '(all)'}")
        print()
        print(f"  {'idx':>4}  source take")
        for index, source in enumerate(manifest.get("episode_ids", ())):
            print(f"  {index:>4}  {source}")
        return 0

    if args.verify:
        store = EpisodeStore(args.episode_root)
        report = verify(args.dataset, store)
        print(format_verify(report))
        print()
        if not report.ok:
            print("refusing to replay a dataset that failed verification.", file=sys.stderr)
            return 2

    try:
        episode = episode_from_dataset(args.dataset, args.episode, so_snake_config=config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    # Task mode is the only meaningful one; see the module docstring.
    replay = ReplayConfig(
        mode="task",
        speed=args.speed,
        realtime=not args.no_realtime,
        check_clearance=not args.no_clearance_check,
    )

    print(f"dataset   {manifest['repo_id']}  episode {args.episode}")
    print(f"          {episode.meta.n_steps} steps, {episode.meta.duration_s:.1f} s "
          f"at {episode.meta.control_hz:g} Hz")
    if episode.meta.notes:
        print(f"          {episode.meta.notes}")
    print(f"replay    task mode at {args.speed:g}x onto {args.backend}")

    try:
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
        print("REAL ARM — it WILL move, along the whole exported trajectory.")
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
