#!/usr/bin/env python
"""What hardware is attached to this machine, and what to pass for it.

Answers the two questions every real-arm command used to start with: which
serial port is the arm on, and which camera index is which. Nothing here moves
the arm -- ports are enumerated, and `--probe` pings servo ids read-only.

    # ports + cameras, with a thumbnail written per camera:
    PYTHONPATH=src .venv/bin/python scripts/scan_devices.py

    # ports only (instant; cameras take about a second per index):
    PYTHONPATH=src .venv/bin/python scripts/scan_devices.py --no-cameras

    # also ping the servo bus on each candidate port (read-only):
    PYTHONPATH=src .venv/bin/python scripts/scan_devices.py --probe

The port is auto-detected by every script that drives the arm, so the usual
reason to run this is the camera thumbnails: on macOS an OpenCV index has no
name and no stable meaning, so the only way to tell two cameras apart is to look
at what each one sees. The images land in `data/device_scan/` and the
`--camera <role>=<device>` flag to paste is printed under each.
"""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from so_snake.config import REPO_ROOT
from so_snake.devices import (
    ARM_PORT_ENV,
    DeviceDetectionError,
    arm_port_candidates,
    detect_arm_port,
    list_serial_ports,
    probe_port,
    scan_cameras,
)
from so_snake.m0_perception import CAMERA_ROLES

DEFAULT_THUMBNAIL_DIR = REPO_ROOT / "data" / "device_scan"


def report_ports(probe: bool) -> str:
    """Print every serial port; return the arm's, or "" if it cannot be told."""
    print("serial ports")
    print("-" * 70)
    ports = list_serial_ports()
    if not ports:
        print("  none enumerated (no adapter attached, or pyserial is not installed)")
        return ""

    candidates = {p.device for p in arm_port_candidates(ports)}
    for p in ports:
        mark = "*" if p.device in candidates else " "
        print(f"  {mark} {p.device:<34} {p.label()}")
        if probe and p.device in candidates:
            ids = probe_port(p.device)
            print(f"      servo ping: {ids if ids else 'no response (unpowered, or not the arm)'}")

    try:
        port = detect_arm_port(probe=probe)
    except DeviceDetectionError as exc:
        print(f"\n  {exc}")
        return ""
    print(f"\n  arm port: {port}")
    print(f"  Scripts detect this themselves; pass --port or set {ARM_PORT_ENV} to override.")
    return port


def report_cameras(directory: Path, max_index: int, write_thumbnails: bool) -> None:
    print("\ncameras")
    print("-" * 70)
    devices = scan_cameras(max_index=max_index, thumbnails=write_thumbnails)
    if not devices:
        print("  none delivered a frame.")
        print("  On macOS: Settings -> Privacy & Security -> Camera must allow this terminal;")
        print("  a denied process is shown no devices rather than an error.")
        return

    if write_thumbnails:
        directory.mkdir(parents=True, exist_ok=True)
    for d in devices:
        print(f"  {d.label()}")
        if write_thumbnails and d.thumbnail.startswith("data:image/jpeg;base64,"):
            path = directory / f"camera_{d.index}.jpg"
            path.write_bytes(base64.b64decode(d.thumbnail.split(",", 1)[1]))
            print(f"      {path}")
        print(f"      --camera {CAMERA_ROLES[0]}={d.device}   (or {CAMERA_ROLES[1]}={d.device})")

    if write_thumbnails:
        print(f"\n  Open {directory} and assign roles by what each camera sees.")
    if len(devices) == 1:
        print("  One camera: --camera <role>=auto resolves to it without a scan listing.")
    else:
        print("  More than one camera, so --camera <role>=auto will refuse: an index is not")
        print("  a name, and a mislabelled view is only noticed by whoever watches the episode.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-cameras", action="store_true", help="skip the (slow) camera scan")
    parser.add_argument("--no-ports", action="store_true", help="skip the serial port scan")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="ping servo ids on each candidate port (read-only; never moves the arm)",
    )
    parser.add_argument("--max-index", type=int, default=8, help="highest camera index to try")
    parser.add_argument("--thumbnail-dir", type=Path, default=DEFAULT_THUMBNAIL_DIR)
    parser.add_argument("--no-thumbnails", action="store_true", help="do not write camera JPEGs")
    args = parser.parse_args()

    if not args.no_ports:
        report_ports(args.probe)
    if not args.no_cameras:
        report_cameras(args.thumbnail_dir, args.max_index, not args.no_thumbnails)
    return 0


if __name__ == "__main__":
    sys.exit(main())
