#!/usr/bin/env python
"""Preflight checks before driving the real SO-100 arm.

Run these first, on the machine the arm is plugged into. Every check but one
touches no hardware; the exception is ``--probe``, which opens the servo bus
*read-only*: it never enables torque and never sends a goal position, so it
cannot move the arm. It only pings the motors and reads their current position.

    # safe, no hardware:
    PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py

    # also ping the servo bus read-only (needs the arm powered + plugged in).
    # The port is auto-detected; --port only to override it:
    PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py --probe

    # and open every camera to see which ones deliver a frame (slow):
    PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py --scan-cameras

Exit code is 0 when nothing FAILed (WARNs are allowed), 1 otherwise.

What this deliberately does NOT do: enable torque, calibrate, or send any
motion. Calibration (moving the arm through its range) and the first commanded
motion are separate, operator-driven steps -- this script only tells you
whether you are ready to take them.
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass, field

from so_snake.config import ARM_JOINTS, GRIPPER_JOINT, SoSnakeConfig
from so_snake.devices import (
    ARM_PORT_ENV,
    DeviceDetectionError,
    arm_port_candidates,
    detect_arm_port,
    list_serial_ports,
    scan_cameras,
)

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_MARK = {PASS: "✓", WARN: "!", FAIL: "✗", SKIP: "-"}

NINTENDO_VENDOR_ID = 0x057E


@dataclass
class Check:
    name: str
    status: str = SKIP
    lines: list[str] = field(default_factory=list)

    def set(self, status: str, *lines: str) -> "Check":
        self.status = status
        self.lines.extend(lines)
        return self


def _imports() -> tuple[dict[str, str], dict[str, str]]:
    """Return (present -> version, missing -> hint)."""
    present: dict[str, str] = {}
    missing: dict[str, str] = {}

    def probe(mod: str, attr: str = "__version__") -> None:
        try:
            m = __import__(mod)
            present[mod] = str(getattr(m, attr, "?"))
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            missing[mod] = f"{type(exc).__name__}: {exc}"

    probe("lerobot")
    probe("scservo_sdk", attr="__name__")  # feetech-servo-sdk; no __version__
    probe("serial")
    probe("hid", attr="__name__")
    probe("numpy")
    return present, missing


def check_platform() -> Check:
    c = Check("platform / python")
    c.set(PASS, f"{platform.platform()}", f"python {sys.version.split()[0]} @ {sys.executable}")
    return c


def check_dependencies() -> Check:
    c = Check("dependencies")
    present, missing = _imports()
    for name, ver in present.items():
        c.lines.append(f"{name:14} {ver}")
    # lerobot must expose the SOFollower robot to drive the arm.
    so_ok = False
    if "lerobot" in present:
        try:
            from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # noqa: F401

            so_ok = True
            c.lines.append("lerobot.robots.so_follower.SOFollower import OK")
        except Exception as exc:  # noqa: BLE001
            missing["lerobot.robots.so_follower"] = f"{type(exc).__name__}: {exc}"

    hard_missing = [m for m in ("lerobot", "scservo_sdk", "serial", "numpy") if m in missing]
    if hard_missing or not so_ok:
        for m in hard_missing:
            c.lines.append(f"MISSING {m}: {missing[m]}")
        if "lerobot.robots.so_follower" in missing:
            c.lines.append(f"MISSING SOFollower: {missing['lerobot.robots.so_follower']}")
        c.lines.append('install with: GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev,sim,teleop]"')
        return c.set(FAIL)
    if "hid" in missing:
        c.lines.append(f"hid (controller input) missing: {missing['hid']}  -- only needed for teleop")
        return c.set(WARN)
    return c.set(PASS)


def check_config() -> Check:
    """Pure check: the arm contract in config matches SOFollower's motor map."""
    c = Check("config / joint contract")
    cfg = SoSnakeConfig()
    expected = (*ARM_JOINTS, GRIPPER_JOINT)

    # Home configuration must be inside the joint limits, or the arm starts by
    # slamming into a limit clamp.
    lo, hi = cfg.arm.limits_deg_array()
    home = cfg.teleop.home_joints_deg
    out = [ARM_JOINTS[i] for i, h in enumerate(home) if not (lo[i] <= h <= hi[i])]
    if out:
        c.set(FAIL, f"home_joints_deg outside limits for: {', '.join(out)}")
    else:
        c.lines.append(f"home_joints_deg {home} within limits")

    # Cross-check against lerobot's own motor map, if importable.
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        robot = SOFollower(SOFollowerRobotConfig(port="/dev/null"))
        bus_motors = {name: m.id for name, m in robot.bus.motors.items()}
        if tuple(bus_motors.keys()) != expected:
            return c.set(
                FAIL,
                f"joint order mismatch: config {expected} vs SOFollower {tuple(bus_motors.keys())}",
            )
        if tuple(bus_motors.values()) != tuple(range(1, len(expected) + 1)):
            return c.set(FAIL, f"unexpected servo ids: {bus_motors}")
        c.lines.append(f"joint order + servo ids match SOFollower: {bus_motors}")
    except Exception as exc:  # noqa: BLE001
        c.lines.append(f"could not cross-check against SOFollower ({type(exc).__name__}: {exc})")
        return c.set(WARN if c.status != FAIL else FAIL)

    return c.set(c.status if c.status == FAIL else PASS)


def check_serial_port(port: str | None) -> tuple[Check, str]:
    """Enumerate the ports, and resolve the one the arm is on.

    Returns the resolved port alongside the check so `--probe` uses exactly the
    port this reported, rather than detecting a second time and possibly
    disagreeing with what the operator just read.
    """
    c = Check("serial port")
    ports = list_serial_ports()
    if not ports:
        return c.set(WARN, "no serial ports enumerated (pyserial missing or none present)"), ""
    candidates = {p.device for p in arm_port_candidates(ports)}
    for p in ports:
        c.lines.append(f"{'* ' if p.device in candidates else '  '}{p.device:<32} {p.label()}")

    try:
        resolved = detect_arm_port(port)
    except DeviceDetectionError as exc:
        return c.set(FAIL, *str(exc).splitlines()[-3:]), ""

    if port:
        if resolved not in {p.device for p in ports}:
            return c.set(FAIL, f"--port {resolved} is not among the enumerated ports above"), resolved
        return c.set(PASS, f"--port {resolved} found"), resolved
    c.lines.append(f"auto-detected arm port: {resolved}  (override with --port or {ARM_PORT_ENV})")
    return c.set(PASS), resolved


def check_controller() -> Check:
    c = Check("Switch Pro controller (teleop input)")
    try:
        import hid
    except Exception as exc:  # noqa: BLE001
        return c.set(WARN, f"hidapi unavailable: {exc}")
    pads = [d for d in hid.enumerate() if d.get("vendor_id") == NINTENDO_VENDOR_ID]
    if not pads:
        return c.set(WARN, "no Nintendo controller detected (only needed for --source pro teleop)")
    for d in pads[:1]:
        c.lines.append(f"found vid=0x{d['vendor_id']:04x} pid=0x{d['product_id']:04x} {d.get('product_string')!r}")
    return c.set(PASS)


def check_cameras(scan: bool) -> Check:
    """List the cameras that actually deliver a frame. Only with --scan-cameras.

    Opt-in because the scan opens every index on the machine (about a second
    each) and on macOS the first open raises the OS permission prompt. Neither
    belongs in a check that is otherwise instant and side-effect free.
    """
    c = Check("cameras")
    if not scan:
        return c.set(SKIP, "not scanned; add --scan-cameras (opens each device, ~1 s per index)")
    devices = scan_cameras()
    if not devices:
        return c.set(
            WARN,
            "no camera delivered a frame.",
            "On macOS check Settings -> Privacy & Security -> Camera for this terminal;",
            "a denied process sees no devices rather than an error.",
        )
    for d in devices:
        c.lines.append(f"{d.label()}{'  (stable path)' if d.stable else ''}")
    c.lines.append("Assign roles by looking at the picture -- the GUI scan shows a thumbnail")
    c.lines.append("of each, and an index is not a name (see so_snake.m0_perception).")
    return c.set(PASS)


def check_calibration(robot_id: str, port: str = "") -> Check:
    c = Check(f"motor calibration (id={robot_id})")
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        robot = SOFollower(SOFollowerRobotConfig(port="/dev/null", id=robot_id))
        fpath = robot.calibration_fpath
    except Exception as exc:  # noqa: BLE001
        return c.set(WARN, f"could not resolve calibration path: {type(exc).__name__}: {exc}")
    c.lines.append(f"path: {fpath}")
    if fpath.is_file():
        return c.set(PASS, "calibration file present")
    c.lines.append("no calibration file -- the arm has not been calibrated under this id.")
    c.lines.append("Calibrate once (moves the arm through its range) before teleop, e.g.:")
    # The detected port goes straight into the command: lerobot-calibrate is an
    # external tool and takes the port, so this is the one place the operator
    # would otherwise still have to look it up.
    c.lines.append(
        f"  lerobot-calibrate --robot.type=so100_follower "
        f"--robot.port={port or '<PORT>'} --robot.id={robot_id}"
    )
    return c.set(WARN, "(WARN, not FAIL: calibration is an expected first-time step)")


def probe_servo_bus(port: str, robot_id: str) -> Check:
    """Read-only servo bus probe. Never enables torque, never commands motion."""
    c = Check("servo bus probe (read-only)")
    try:
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    except Exception as exc:  # noqa: BLE001
        return c.set(FAIL, f"cannot import SOFollower: {exc}")

    robot = SOFollower(SOFollowerRobotConfig(port=port, id=robot_id))
    bus = robot.bus
    try:
        bus.connect()  # opens the port and pings motors; no torque, no motion
    except Exception as exc:  # noqa: BLE001
        return c.set(
            FAIL,
            f"bus.connect failed on {port}: {type(exc).__name__}: {exc}",
            "check the arm is powered, the USB cable is seated, and --port is right.",
        )
    try:
        responded, missing = [], []
        for name in bus.motors:
            model = bus.ping(name, raise_on_error=False)
            (responded if model is not None else missing).append(name)
        c.lines.append(f"responded: {responded}")
        if missing:
            c.lines.append(f"NO RESPONSE: {missing}  (wiring / servo id / power)")
        # Raw present position (0-4095 ticks); no calibration needed, no motion.
        try:
            raw = bus.sync_read("Present_Position", normalize=False)
            c.lines.append("Present_Position (raw ticks): " + ", ".join(f"{k}={v}" for k, v in raw.items()))
        except Exception as exc:  # noqa: BLE001
            c.lines.append(f"could not read Present_Position: {type(exc).__name__}: {exc}")
        status = FAIL if missing or len(responded) != len(bus.motors) else PASS
        if status == PASS:
            c.lines.append("all 6 servos present; positions read without moving the arm.")
        return c.set(status)
    finally:
        # Preserve torque state exactly as found -- do not risk the arm sagging.
        try:
            bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--port",
        default=None,
        help="arm serial port; auto-detected when omitted (e.g. /dev/cu.usbmodem58760434321)",
    )
    parser.add_argument("--id", default="so_snake", help="lerobot robot id (calibration file name)")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="open the servo bus read-only and ping motors (never moves the arm)",
    )
    parser.add_argument(
        "--scan-cameras",
        action="store_true",
        help="open every camera index to see which ones deliver a frame (~1 s each)",
    )
    args = parser.parse_args()

    port_check, port = check_serial_port(args.port)
    checks = [
        check_platform(),
        check_dependencies(),
        check_config(),
        port_check,
        check_controller(),
        check_cameras(args.scan_cameras),
        check_calibration(args.id, port),
    ]

    if args.probe:
        if not port:
            checks.append(
                Check("servo bus probe (read-only)").set(FAIL, "no arm port; see the serial port check")
            )
        elif any(c.status == FAIL for c in checks[:2]):
            checks.append(Check("servo bus probe (read-only)").set(SKIP, "skipped: fix dependencies first"))
        else:
            checks.append(probe_servo_bus(port, args.id))
    else:
        checks.append(
            Check("servo bus probe (read-only)").set(
                SKIP, "not run; add --probe to ping the servos (read-only)"
            )
        )

    print("=" * 70)
    print("SO-100 real-arm preflight")
    print("=" * 70)
    for c in checks:
        print(f"[{_MARK[c.status]}] {c.status:4} {c.name}")
        for line in c.lines:
            print(f"        {line}")
    print("=" * 70)

    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    verdict = "FAIL" if n_fail else ("PASS with warnings" if n_warn else "PASS")
    print(f"RESULT: {verdict}   ({n_fail} fail, {n_warn} warn)")
    if not n_fail:
        print("Next: calibrate if needed, then a slow first teleop with a low --max-relative-target.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
