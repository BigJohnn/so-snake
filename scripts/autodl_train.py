#!/usr/bin/env python
"""SSH/rsync bridge used by the GUI for an AutoDL training run.

It deliberately assumes SSH keys or an ssh-agent are already configured.  No
password, W&B token, or cloud API key crosses the GUI or lands in this repo.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run(argv: list[str]) -> None:
    print("+ " + shlex.join(argv), flush=True)
    subprocess.run(argv, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--host", required=True); parser.add_argument("--user", required=True)
    parser.add_argument("--port", required=True); parser.add_argument("--remote-root", required=True)
    parser.add_argument("--remote-python", required=True); parser.add_argument("--run-name", required=True)
    parser.add_argument("train_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.train_argv or args.train_argv[0] != "--":
        raise ValueError("expected '--' before the training command")
    remote = f"{args.user}@{args.host}"; ssh = ["ssh", "-p", args.port, remote]
    remote_run = f"{args.remote_root}/so-snake-runs/{args.run_name}"
    remote_data = f"{remote_run}/data/{args.dataset.name}"
    remote_output = f"{remote_run}/outputs/{args.run_name}"
    run([*ssh, "mkdir", "-p", remote_data, remote_output])
    run(["rsync", "-az", "--info=progress2", "-e", f"ssh -p {args.port}", f"{args.dataset}/", f"{remote}:{remote_data}/"])
    command = [args.remote_python, *args.train_argv[1:]]
    command = [part.replace(str(args.dataset), remote_data).replace(str(args.local_output), remote_output) for part in command]
    run([*ssh, "bash", "-lc", f"cd {shlex.quote(remote_run)} && {shlex.join(command)}"])
    args.local_output.parent.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-az", "--info=progress2", "-e", f"ssh -p {args.port}", f"{remote}:{remote_output}/", f"{args.local_output}/"])
    print(f"checkpoint copied to {args.local_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
