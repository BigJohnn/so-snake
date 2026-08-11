"""Finding the hardware instead of being told where it is.

Two device families have to be named before anything can run: the arm's USB
serial port and the cameras. Both names are unstable -- the port is
`/dev/cu.usbmodem58760434321` on this bench and `/dev/ttyACM0` on the next
machine, and it changes when the adapter lands on a different USB port -- so
every script that took a mandatory `--port` was asking the operator to look it
up first. This module does the looking up.

**The port is identified, not guessed.** The SO-100's servo driver board is a
USB serial adapter with a known chip (a WCH CH343 here, `1a86:55d3`), so
candidates are scored by USB vendor/product id first and by device-name shape
only as a fallback. Ports that are never an arm -- the macOS Bluetooth serial
port, the debug console -- are excluded outright, because a "likely port"
heuristic that offers `/dev/cu.Bluetooth-Incoming-Port` and then hangs opening
it is worse than no heuristic.

**One candidate is answered without touching the bus.** The common case (one
arm plugged in) resolves from enumeration alone: no port is opened, nothing is
written, and detection cannot disturb a session. Only genuine ambiguity -- two
or more USB serial adapters present -- falls back to `probe_port`, which opens
each candidate read-only and pings servo ids. Pinging asks a servo for its model
number; it does not enable torque and cannot move the arm.

**Cameras are enumerated, never named by inference.** `so_snake.m0_perception`
explains at length why an OpenCV index cannot be mapped to a device name on
macOS, and nothing here walks that back: `scan_cameras` is a thin, typed pass
over `list_devices`. What `resolve_camera_specs` adds is the one case where a
role *can* be filled without the operator looking at a thumbnail -- exactly one
unclaimed camera exists, so there is nothing to get wrong. Two candidates and it
refuses and lists them, because assigning the wrist role to the third-person
camera produces episodes that look fine until someone trains on them.
"""

from __future__ import annotations

import glob
import os
import platform
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .m0_perception import CAMERA_ROLES, CameraSpec, list_devices

# Environment override, honoured by every `detect_arm_port` caller. The point is
# a bench where detection is wrong (two arms, an adapter this table has never
# seen) can be fixed once for the shell instead of on every command line.
ARM_PORT_ENV = "SO_SNAKE_ARM_PORT"

# "the operator did not name a port". Both spellings arrive: "" from an argparse
# default and a JSON body, "auto" from someone being explicit about it.
AUTO = ("", "auto", "detect")

# USB vendor:product ids of the serial bridges used by SO-100/SO-101 driver
# boards and the common breakout adapters. A match here is near-certain: these
# chips are on the board, not on the laptop.
KNOWN_USB_IDS: dict[tuple[int, int], str] = {
    (0x1A86, 0x55D3): "WCH CH343 (SO-100/101 driver board)",
    (0x1A86, 0x7523): "WCH CH340",
    (0x1A86, 0x7522): "WCH CH340",
    (0x0403, 0x6001): "FTDI FT232R",
    (0x0403, 0x6014): "FTDI FT232H",
    (0x10C4, 0xEA60): "Silicon Labs CP210x",
    (0x067B, 0x2303): "Prolific PL2303",
}

# Vendors whose bridges show up on these boards but whose product ids vary.
KNOWN_USB_VENDORS: dict[int, str] = {0x1A86: "WCH", 0x0403: "FTDI", 0x10C4: "Silicon Labs"}

# Device-name fragments that mean "USB serial adapter" on some machine. Weaker
# evidence than a USB id -- this is what identifies a port on a Linux box where
# pyserial reports no vid/pid -- but strong enough to shortlist.
_NAME_HINTS = ("usbmodem", "usbserial", "ttyacm", "ttyusb", "wchusbserial", "ftdi", "ch340")

# Never the arm. `Bluetooth-Incoming-Port` in particular is present on every
# macOS machine and blocks for seconds when opened.
_EXCLUDED = ("bluetooth", "debug-console", "wlan-debug", "console", "irda")

# The Feetech bus the SO-100 runs, and the ids its six servos answer on.
SERVO_BAUDRATE = 1_000_000
SERVO_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


class DeviceDetectionError(RuntimeError):
    """No device could be identified, or several could and none is the obvious one."""


# ------------------------------------------------------------- serial ports


@dataclass(frozen=True)
class SerialPortInfo:
    """One enumerated serial port, and what the USB layer says about it."""

    device: str
    description: str = ""
    vid: int | None = None
    pid: int | None = None
    serial_number: str = ""
    manufacturer: str = ""

    @property
    def usb_id(self) -> str:
        """`"1a86:55d3"`, or "" when the port is not a USB device."""
        if self.vid is None or self.pid is None:
            return ""
        return f"{self.vid:04x}:{self.pid:04x}"

    @property
    def known_as(self) -> str:
        """The name of the bridge chip when it is one we recognise, else ""."""
        if self.vid is None or self.pid is None:
            return ""
        return KNOWN_USB_IDS.get((self.vid, self.pid), KNOWN_USB_VENDORS.get(self.vid, ""))

    @property
    def score(self) -> int:
        """How likely this port is to be the arm. Negative means "never"."""
        name = self.device.lower()
        if any(bad in name for bad in _EXCLUDED):
            return -1
        if self.vid is not None and self.pid is not None:
            if (self.vid, self.pid) in KNOWN_USB_IDS:
                return 100
            if self.vid in KNOWN_USB_VENDORS:
                return 80
        if any(hint in name for hint in _NAME_HINTS):
            # A USB serial device with an id we do not know, or a platform that
            # does not report ids at all. Still a candidate; just not a match.
            return 50 if self.vid is not None else 40
        return 0

    def label(self) -> str:
        # pyserial says "n/a" rather than nothing for a port with no USB layer
        # behind it; printing that back at the operator is just noise.
        description = "" if self.description.strip().lower() == "n/a" else self.description.strip()
        bits = [b for b in (description, self.known_as, self.usb_id) if b]
        # The description already repeats the chip name on some platforms.
        seen: list[str] = []
        for bit in bits:
            if bit not in seen:
                seen.append(bit)
        return "  ".join(seen)


def list_serial_ports() -> list[SerialPortInfo]:
    """Every serial port this machine reports, in enumeration order.

    Falls back to globbing `/dev` when pyserial is missing, because the offline
    gates install neither lerobot nor pyserial and a scan script that raises
    ImportError is less useful than one that lists device nodes without USB
    metadata.
    """
    try:
        from serial.tools import list_ports
    except Exception:  # noqa: BLE001 - pyserial absent; degrade, do not raise
        return [
            SerialPortInfo(device=path)
            for pattern in ("/dev/cu.usb*", "/dev/ttyACM*", "/dev/ttyUSB*")
            for path in sorted(glob.glob(pattern))
        ]

    ports: list[SerialPortInfo] = []
    for p in list_ports.comports():
        # macOS exposes every port twice, as /dev/tty.* (waits for carrier
        # detect on open) and /dev/cu.* (does not). Only cu.* is usable for a
        # servo bus; pyserial normally reports only that one, but the glob
        # fallback and other platforms make the guard worth keeping.
        if platform.system() == "Darwin" and p.device.startswith("/dev/tty."):
            continue
        ports.append(
            SerialPortInfo(
                device=p.device,
                description=(p.description or "").strip(),
                vid=p.vid,
                pid=p.pid,
                serial_number=(p.serial_number or "").strip(),
                manufacturer=(p.manufacturer or "").strip(),
            )
        )
    return ports


def arm_port_candidates(ports: Sequence[SerialPortInfo] | None = None) -> list[SerialPortInfo]:
    """The ports that could be the arm, best first. Excludes the never-the-arm ones."""
    ports = list_serial_ports() if ports is None else ports
    candidates = [p for p in ports if p.score > 0]
    # Stable within a score: enumeration order is the operator's plug order on
    # Linux, so detection does not flip between runs of the same setup.
    return sorted(candidates, key=lambda p: -p.score)


def probe_port(
    device: str,
    *,
    baudrate: int = SERVO_BAUDRATE,
    ids: Iterable[int] = (1, 2, 3),
) -> list[int]:
    """Servo ids that answer on `device`. Read-only: pings, never writes.

    A ping asks a servo for its model number. It does not enable torque, does
    not send a goal position, and cannot move the arm -- the same guarantee
    `scripts/preflight_real_arm.py --probe` makes, which is what lets detection
    fall back to this when enumeration alone is ambiguous.

    Returns `[]` for every not-this-one case (port busy, wrong device, arm
    unpowered), because the caller's question is only ever "is the arm here".

    Three ids rather than all six: a bus that answers at all answers on the
    first, and the ids only continue so that a single dead servo does not make
    the arm invisible. Each unanswered ping costs the SDK's own packet timeout
    -- it sets that itself inside `ping`, so there is nothing to tune here --
    which is why this is the fallback and not the first thing tried.
    """
    try:
        from scservo_sdk import PacketHandler, PortHandler
    except Exception:  # noqa: BLE001 - no SDK, no probe; enumeration still works
        return []

    port = PortHandler(device)
    try:
        if not port.openPort() or not port.setBaudRate(baudrate):
            return []
        packet = PacketHandler(0)  # protocol_end=0: STS/SMS, what the SO-100 uses
        found: list[int] = []
        for scs_id in ids:
            try:
                _model, comm, error = packet.ping(port, scs_id)
            except Exception:  # noqa: BLE001 - a silent bus raises here on some SDKs
                continue
            if comm == 0 and error == 0:
                found.append(int(scs_id))
        return found
    except Exception:  # noqa: BLE001 - unopenable port is "not the arm", not a crash
        return []
    finally:
        try:
            port.closePort()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass


def detect_arm_port(port: str | None = None, *, probe: bool = True) -> str:
    """The arm's serial port. Explicit beats environment beats detection.

    `port` is returned untouched when it names something, so this can be called
    unconditionally by any script whose `--port` is now optional. Detection
    proper runs only for "", "auto", or None:

      * exactly one candidate -> that one, without opening anything;
      * several -> ping each and take the one whose servos answer;
      * none, or still ambiguous -> `DeviceDetectionError` listing what was
        found and naming the flag to pass.

    Raises rather than returning "" so a caller cannot accidentally hand an
    empty port to lerobot, which reports it as a confusing open failure.
    """
    if port is not None and port.strip().lower() not in AUTO:
        return port.strip()

    override = os.environ.get(ARM_PORT_ENV, "").strip()
    if override and override.lower() not in AUTO:
        return override

    ports = list_serial_ports()
    candidates = arm_port_candidates(ports)
    if not candidates:
        raise DeviceDetectionError(
            "no serial port that looks like the arm.\n"
            + _port_listing(ports)
            + "\nCheck the USB cable and that the driver board is powered, "
            f"or pass --port / set {ARM_PORT_ENV}."
        )
    if len(candidates) == 1:
        return candidates[0].device

    if probe:
        answered = [(p, probe_port(p.device)) for p in candidates]
        alive = [p for p, ids in answered if ids]
        if len(alive) == 1:
            return alive[0].device
        if len(alive) > 1:
            raise DeviceDetectionError(
                "several ports have Feetech servos answering on them:\n"
                + _port_listing([p for p in alive])
                + "\nPass --port to say which arm to drive."
            )

    raise DeviceDetectionError(
        f"{len(candidates)} serial ports could be the arm and none identified itself"
        f"{' (no servo answered a ping)' if probe else ''}:\n"
        + _port_listing(candidates)
        + f"\nPass --port to choose one, or set {ARM_PORT_ENV}."
    )


def describe_arm_port(port: str | None = None, *, probe: bool = True) -> str:
    """`detect_arm_port` plus how it got there, for a line of script output."""
    if port is not None and port.strip().lower() not in AUTO:
        return f"{port.strip()}  (given)"
    if os.environ.get(ARM_PORT_ENV, "").strip():
        return f"{detect_arm_port(port, probe=probe)}  (from {ARM_PORT_ENV})"
    device = detect_arm_port(port, probe=probe)
    info = next((p for p in list_serial_ports() if p.device == device), None)
    detail = f" -- {info.label()}" if info and info.label() else ""
    return f"{device}  (auto-detected{detail})"


def _port_listing(ports: Sequence[SerialPortInfo]) -> str:
    if not ports:
        return "  (no serial ports at all)"
    return "\n".join(f"  {p.device:<32} {p.label()}" for p in ports)


# ------------------------------------------------------------------ cameras


@dataclass(frozen=True)
class CameraDevice:
    """One camera that actually delivered a frame during a scan."""

    index: int
    device: int | str
    width: int
    height: int
    name: str = ""
    stable: bool = False
    bus: str = ""
    thumbnail: str = ""

    def label(self) -> str:
        bits = [f"index {self.index}", f"{self.width}x{self.height}"]
        if self.name:
            bits.append(self.name)
        if self.stable:
            bits.append(str(self.device))
        return "  ".join(bits)


def scan_cameras(max_index: int = 8, thumbnails: bool = False) -> list[CameraDevice]:
    """Cameras attached to this machine, one open-and-read per index.

    Slow by nature -- roughly a second per device, and on macOS the first call
    raises the OS permission prompt -- so this belongs behind an explicit
    request, never in a poll. Thumbnails are off by default here (the CLI wants
    a list, the GUI wants pictures) and cost an extra JPEG encode each.
    """
    return [
        CameraDevice(
            index=int(d["index"]),
            device=d["device"],
            width=int(d["width"]),
            height=int(d["height"]),
            name=str(d.get("name", "")),
            stable=bool(d.get("stable", False)),
            bus=str(d.get("bus", "")),
            thumbnail=str(d.get("thumbnail", "")),
        )
        for d in list_devices(max_index=max_index, thumbnails=thumbnails)
    ]


def resolve_camera_specs(
    assignments: dict[str, Any],
    devices: Sequence[CameraDevice] | None = None,
) -> tuple[CameraSpec, ...]:
    """Turn role -> device (or "auto") into `CameraSpec`s, scanning only if asked.

    An explicit index or path passes straight through; no scan happens, so a
    fully specified rig costs nothing. `"auto"` is filled from a scan, and only
    when the answer is not a guess: exactly one camera that no other role has
    claimed. Anything else raises with the scan listed, because the failure mode
    of guessing here is silent -- a wrist-labelled third-person view is only
    discovered by whoever watches the episode later.
    """
    wanted = {role: value for role, value in assignments.items() if value not in (None, "")}
    for role in wanted:
        if role not in CAMERA_ROLES:
            raise ValueError(f"camera role must be one of {CAMERA_ROLES}, got {role!r}")

    explicit = {
        role: (int(v) if isinstance(v, str) and v.strip().lstrip("-").isdigit() else v)
        for role, v in wanted.items()
        if not (isinstance(v, str) and v.strip().lower() in AUTO)
    }
    auto_roles = [role for role in wanted if role not in explicit]
    if not auto_roles:
        return tuple(CameraSpec(role=r, index_or_path=v) for r, v in explicit.items())

    found = list(scan_cameras()) if devices is None else list(devices)
    claimed = {str(v) for v in explicit.values()}
    free = [d for d in found if str(d.device) not in claimed and str(d.index) not in claimed]

    resolved = dict(explicit)
    for role in auto_roles:
        if not free:
            raise DeviceDetectionError(
                f"camera role {role!r} is set to auto but no unassigned camera was found.\n"
                + _camera_listing(found)
            )
        if len(free) > 1:
            raise DeviceDetectionError(
                f"camera role {role!r} is set to auto but {len(free)} cameras are "
                "available and nothing can say which is which:\n"
                + _camera_listing(free)
                + "\nName the device instead, e.g. --camera "
                f"{role}={free[0].device}. The GUI shows a thumbnail of each."
            )
        resolved[role] = free.pop().device

    # Role order follows CAMERA_ROLES so the rig is built the same way whichever
    # order the flags were typed in.
    return tuple(
        CameraSpec(role=role, index_or_path=resolved[role])
        for role in CAMERA_ROLES
        if role in resolved
    )


def _camera_listing(devices: Sequence[CameraDevice]) -> str:
    if not devices:
        return "  (no cameras delivered a frame)"
    return "\n".join(f"  {d.label()}" for d in devices)
