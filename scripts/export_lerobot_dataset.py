#!/usr/bin/env python
"""Export recorded episodes into a `LeRobotDataset` for ACT training.

Selects by task label, so one store holding several skills exports one skill at
a time — a policy trained across two tasks learns their average.

    PYTHONPATH=src python scripts/export_lerobot_dataset.py --list-tasks
    PYTHONPATH=src python scripts/export_lerobot_dataset.py --task "牛牛抓放" --dry-run
    PYTHONPATH=src python scripts/export_lerobot_dataset.py --task "牛牛抓放" \\
        --repo-id so_snake/niuniu_pick_place

State is the 5D manifold pose the arm *reached* plus the gripper; action is a
step along that manifold anchored on the reached pose, plus an absolute gripper.
`--action-space absolute` exports the target itself instead, as the control for
diagnosing a drifting rollout. The reasoning behind both, and behind the frame
rate being measured rather than read from the config, is in
`src/so_snake/data/export.py`.

`--dry-run` screens the episodes, measures the rate and prints the action
statistics without writing a dataset. Run it first: it is the cheapest place to
find out that a take lost its video alignment.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from so_snake.data import DEFAULT_EPISODE_ROOT, EpisodeStore
from so_snake.data.export import (
    ACTION_SPACES,
    ExportConfig,
    export,
    format_report,
    format_verify,
    plan,
    verify,
    validate_roi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_EPISODE_ROOT,
                        help="episode store to read from")
    parser.add_argument("--list-tasks", action="store_true",
                        help="show the task labels in the store, with counts, and exit")
    parser.add_argument("--task", default=None,
                        help="exact task label to export; omit to take every episode")
    parser.add_argument("--episode", action="append", default=[], dest="episode_ids",
                        help="export these ids instead of selecting by task; repeatable")
    parser.add_argument("--repo-id", default="so_snake/export",
                        help="LeRobotDataset repo id")
    parser.add_argument("--out", type=Path, default=None,
                        help="dataset root; defaults to the lerobot cache")
    parser.add_argument("--action-space", choices=ACTION_SPACES, default="delta")
    parser.add_argument("--camera", action="append", default=[], dest="cameras",
                        help="camera role to include; repeatable "
                             "(default: third_person and wrist)")
    parser.add_argument("--resolution", default="240x320", metavar="HxW",
                        help="frame size to resize to (default 240x320, which trains "
                             "at ~300 ms/step on an M1 Pro)")
    parser.add_argument("--fps", type=int, default=None,
                        help="dataset frame rate. A lower integer divisor (e.g. 60 -> 30) "
                             "samples state/action/video together; other rates are refused")
    parser.add_argument("--roi", action="append", default=[], metavar="ROLE=X,Y,W,H",
                        help="normalised image crop; repeat per camera, and it is recorded "
                             "for matching rollout preprocessing")
    parser.add_argument("--include-aborted", action="store_true",
                        help="also export takes that ended for a reason other than "
                             "the operator stopping them")
    parser.add_argument("--dry-run", action="store_true",
                        help="screen and report; write nothing")
    parser.add_argument("--verify", type=Path, default=None, metavar="DATASET",
                        help="read an already-exported dataset back off disk and check "
                             "it replays to what was recorded, then exit")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the read-back check that normally follows an export")
    parser.add_argument("--overwrite", action="store_true",
                        help="wipe the target directory before exporting. Destructive: "
                             "the existing parquet, videos and manifest are deleted. "
                             "Use when re-running an export into a directory whose "
                             "contents should be replaced; refuse without the flag "
                             "otherwise, so an unrelated dataset cannot be overwritten "
                             "by mistake.")
    return parser.parse_args()


def list_tasks(store: EpisodeStore) -> int:
    metas = store.list_meta()
    if not metas:
        print(f"no episodes under {store.root}")
        return 1
    counts = Counter(m.task for m in metas)
    print(f"{len(metas)} episodes under {store.root}")
    print(f"  {'steps':>6}  {'takes':>5}  task")
    for task, n in counts.most_common():
        steps = sum(m.n_steps for m in metas if m.task == task)
        print(f"  {steps:6d}  {n:5d}  {task or '(unlabelled)'}")
    return 0


def main() -> int:
    args = parse_args()
    store = EpisodeStore(args.root)

    if args.list_tasks:
        return list_tasks(store)

    if args.verify is not None:
        try:
            report = verify(args.verify, store)
        except (FileNotFoundError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1
        print(format_verify(report))
        return 0 if report.ok else 1

    try:
        height, width = (int(v) for v in args.resolution.lower().split("x"))
    except ValueError:
        print(f"--resolution must look like HxW, got {args.resolution!r}", file=sys.stderr)
        return 2

    roi = {}
    for item in args.roi:
        role, sep, values = item.partition("=")
        if not sep or not role:
            print(f"--roi must look like ROLE=X,Y,W,H, got {item!r}", file=sys.stderr)
            return 2
        try:
            roi[role] = validate_roi(values.split(","), label=f"ROI for {role}")
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

    config = ExportConfig(
        repo_id=args.repo_id,
        root=args.out,
        task=args.task,
        episode_ids=tuple(args.episode_ids),
        action_space=args.action_space,
        cameras=tuple(args.cameras) if args.cameras else ("third_person", "wrist"),
        resolution=(height, width),
        fps=args.fps,
        roi=roi,
        include_aborted=args.include_aborted,
        episode_root=args.root,
    )

    if args.task is None and not args.episode_ids:
        print("warning: exporting every task in the store into one dataset; "
              "pass --task to train on a single skill", file=sys.stderr)

    def progress(episode_id: str, n: int, done: int, total: int) -> None:
        print(f"  [{done}/{total}] {episode_id}: {n} frames", flush=True)

    try:
        if args.dry_run:
            report, usable = plan(store, config)
            print("DRY RUN — nothing written")
            print(format_report(report, config))
            return 0 if usable else 1
        report = export(store, config, progress=progress, overwrite=args.overwrite)
    except (ValueError, FileExistsError) as error:
        # Both end up on stderr and exit non-zero: a 4xx-shaped refusal (the
        # caller can fix it) reads the same as a 5xx-shaped one if the exit
        # code or stream differs, and both deserve to.
        print(str(error), file=sys.stderr)
        return 1
    print(format_report(report, config))

    if args.no_verify or report.dataset_path is None:
        return 0
    # Verifying by default, because an export nobody read back is a claim, not a
    # result: the failures that matter here -- an unwritten parquet footer, a
    # video a frame short, a time axis built from a rate nothing ran at -- all
    # look like success at the moment of writing.
    print()
    verified = verify(report.dataset_path, store)
    print(format_verify(verified))
    return 0 if verified.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
