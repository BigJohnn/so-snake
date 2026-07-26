#!/usr/bin/env python
"""Preflight checks before driving the real SO-100 arm.

Run these first, on the machine the arm is plugged into. Every check but one
touches no hardware; the exception is ``--probe``, which opens the servo bus
*read-only*: it never enables torque and never sends a goal position, so it
cannot move the arm. It only pings the motors and reads their current position.

    # safe, no hardware:
    PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py

    # also ping the servo bus read-only (needs the arm powered + plugged in):
    PYTHONPATH=src .venv/bin/python scripts/preflight_real_arm.py \
        --port /dev/cu.usbmodem58760434321 --probe

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

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_MARK = {PASS: "✓", WARN: "!", FAIL: "✗", SKIP: "-"}

# Serial-port name fragments that usually denote a USB serial adapter (the arm),
# as opposed to a Bluetooth or debug console port.
_LIKELY_PORT_HINTS = ("usbmodem", "usbserial", "ttyacm", "ttyusb", "ftdi", "ch340", "wch")
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


def _list_ports() -> list[tuple[str, str]]:
    try:
        from serial.tools import list_ports
    except Exception:  # noqa: BLE001
        return []
    return [(p.device, (p.description or "").strip()) for p in list_ports.comports()]


def check_serial_port(port: str | None) -> Check:
    c = Check("serial port")
    ports = _list_ports()
    if not ports:
        return c.set(WARN, "no serial ports enumerated (pyserial missing or none present)")
    for dev, desc in ports:
        likely = any(h in dev.lower() for h in _LIKELY_PORT_HINTS)
        c.lines.append(f"{'* ' if likely else '  '}{dev}    {desc}")
    likely_ports = [dev for dev, _ in ports if any(h in dev.lower() for h in _LIKELY_PORT_HINTS)]

    if port is None:
        if likely_ports:
            c.lines.append(f"likely arm port: {likely_ports[0]}  (pass --port to select)")
        return c.set(WARN, "no --port given; skipping port validation")
    if port not in [dev for dev, _ in ports]:
        return c.set(FAIL, f"--port {port} is not among the enumerated ports above")
    c.lines.append(f"--port {port} found")
    return c.set(PASS)


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


def check_calibration(robot_id: str) -> Check:
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
    c.lines.append(
        f"  lerobot-calibrate --robot.type=so100_follower "
        f"--robot.port=<PORT> --robot.id={robot_id}"
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
    parser.add_argument("--port", default=None, help="arm serial port, e.g. /dev/cu.usbmodem58760434321")
    parser.add_argument("--id", default="so_snake", help="lerobot robot id (calibration file name)")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="open the servo bus read-only and ping motors (needs --port; never moves the arm)",
    )
    args = parser.parse_args()

    checks = [
        check_platform(),
        check_dependencies(),
        check_config(),
        check_serial_port(args.port),
        check_controller(),
        check_calibration(args.id),
    ]

    if args.probe:
        if not args.port:
            checks.append(Check("servo bus probe (read-only)").set(FAIL, "--probe requires --port"))
        elif any(c.status == FAIL for c in checks[:2]):
            checks.append(Check("servo bus probe (read-only)").set(SKIP, "skipped: fix dependencies first"))
        else:
            checks.append(probe_servo_bus(args.port, args.id))
    else:
        checks.append(
            Check("servo bus probe (read-only)").set(
                SKIP, "not run; add --probe --port <PORT> to ping the servos (read-only)"
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
