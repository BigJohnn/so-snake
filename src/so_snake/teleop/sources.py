"""Teleoperation input: the device contract, and the two things that satisfy it.

`NintendoProSample` is deliberately raw and robot-agnostic — timestamps, stick
positions, an absolute IMU quaternion, the clutch, the gripper — and nothing
else. That is plan step 5, and the reason is a boundary rather than a taste:

    the device layer does not know how many orientation degrees of freedom the
    robot has, which of them are controllable, or where the TCP is, so it must
    not be the layer that decides.

The old `TeleopCommand` violated that. It carried `delta_rpy` and
`absolute_rpy`, which are statements about a 6-DoF end effector, so a controller
driver was quietly asserting the arm's action space. Everything downstream then
had to work around the assertion. Now the device reports what the device
measured, `ClutchRetargeter` decides what it means for this arm, and swapping
the controller changes one file.

It also makes the recorded dataset auditable. `action.raw.*` is exactly this
structure, so the chain *raw intent -> projected intent -> executed joints* can
be replayed offline against a different projector or a different IK without
re-recording anything.

Two implementations:

  NintendoProSource  the real Switch Pro controller, via lerobot's NintendoTeleop
  ScriptedSource     a deterministic recorded trajectory, no hardware

The scripted source is not a toy. With neither controller nor arm in hand it is
the only way to exercise the loop end to end, and being deterministic it makes
loop behaviour testable in CI in a way that waggling a real stick never could.

Note on why we do not use joycon-robotics directly: its `JoyconRobotics` class
raises `IOError` unless the device serial matches a specific vendor's Joy-Con,
so it will not open a stock Switch Pro controller. We take lerobot's device
layer, which does support `NintendoController.PRO`, and keep joycon-robotics'
*architecture* — absolute target, workspace clamp, absolute IMU orientation.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class NintendoProSample:
    """One frame from the controller. Raw, and robot-agnostic by construction."""

    t: float  # seconds since the source connected

    # Stick deflections in [-1, 1]. Which stick drives which world axis is a
    # mapping decision, and mappings do not live here.
    left_stick: np.ndarray  # (2,) x, y
    right_stick: np.ndarray  # (2,) x, y

    # Absolute attitude of the controller as (w, x, y, z). Absolute, not
    # integrated: joycon-robotics drives orientation from the attitude estimate
    # because integrating gyro rates accumulates drift and an attitude estimate
    # does not.
    imu_quaternion: np.ndarray  # (4,)

    # True while the operator holds the clutch. Motion is commanded only then;
    # releasing freezes the target so they can reposition their hands.
    clutch: bool = False

    # Gripper opening in [0, 1]; 0 closed, 1 open.
    gripper: float = 1.0

    # Free-form edge-triggered events: "stop", "reset", "next_episode", ...
    events: frozenset[str] = frozenset()

    def as_log_record(self) -> dict[str, np.ndarray | float | bool]:
        """The `action.raw.*` half of the dataset layout."""
        return {
            "action.raw.t": self.t,
            "action.raw.sticks": np.concatenate([self.left_stick, self.right_stick]),
            "action.raw.imu_quaternion": np.asarray(self.imu_quaternion, float),
            "action.raw.clutch": bool(self.clutch),
            "action.raw.gripper": float(self.gripper),
        }


@runtime_checkable
class TeleopSource(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    @property
    def is_connected(self) -> bool: ...

    def read(self) -> NintendoProSample: ...


@dataclass
class ScriptedSource:
    """Replays a fixed sequence of samples, then reports exhaustion.

    Use `from_waveform` for a smooth sweep that exercises the workspace, or pass
    an explicit list to reproduce a specific situation in a test.
    """

    samples: list[NintendoProSample]
    loop: bool = False

    _index: int = field(default=0, init=False)
    _connected: bool = field(default=False, init=False)

    @property
    def exhausted(self) -> bool:
        return not self.loop and self._index >= len(self.samples)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read(self) -> NintendoProSample:
        if not self._connected:
            raise RuntimeError("ScriptedSource is not connected; call connect() first")
        if self._index >= len(self.samples):
            if not self.loop:
                return NintendoProSample(
                    t=float(self._index),
                    left_stick=np.zeros(2),
                    right_stick=np.zeros(2),
                    imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                    clutch=False,
                    events=frozenset({"stop"}),
                )
            self._index = 0
        sample = self.samples[self._index]
        self._index += 1
        return sample

    @classmethod
    def from_waveform(
        cls,
        n_steps: int,
        *,
        amplitude: float = 1.0,
        rotation_amplitude_rad: float = 0.10,
        clutch: bool = True,
        gripper_fn: Callable[[int], float] | None = None,
    ) -> ScriptedSource:
        """A sweep with mutually incommensurate periods on every axis.

        No two axes ever repeat in phase, so the target walks a wide swathe of
        the workspace instead of retracing one path — including into the clamp
        walls, which is where the interesting failures are.

        The IMU follows a smooth attitude rather than a rotation rate, since
        that is what the real controller reports and the difference matters:
        an attitude that returns to where it started must leave the arm's
        orientation where it started too.
        """
        samples = []
        for i in range(n_steps):
            t = i / max(n_steps - 1, 1)
            left = amplitude * np.array(
                [np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 1.7 * t)]
            )
            right = amplitude * np.array([np.sin(2 * np.pi * 2.3 * t), 0.0])

            # A small attitude wandering over all three axes, as a quaternion.
            # Keep the default mild: this is a loop-health sweep, while larger
            # values intentionally stress the atlas and joint-rate walls.
            angles = rotation_amplitude_rad * np.array(
                [
                    np.sin(2 * np.pi * 0.7 * t),
                    np.sin(2 * np.pi * 1.3 * t),
                    np.sin(2 * np.pi * 0.5 * t),
                ]
            )
            half = angles / 2.0
            quaternion = np.array(
                [
                    np.cos(half).prod(),
                    np.sin(half[0]) * np.cos(half[1]) * np.cos(half[2]),
                    np.cos(half[0]) * np.sin(half[1]) * np.cos(half[2]),
                    np.cos(half[0]) * np.cos(half[1]) * np.sin(half[2]),
                ]
            )
            samples.append(
                NintendoProSample(
                    t=i / 30.0,
                    left_stick=left,
                    right_stick=right,
                    imu_quaternion=quaternion,
                    clutch=clutch,
                    gripper=gripper_fn(i) if gripper_fn else (1.0 if (i // 40) % 2 == 0 else 0.0),
                )
            )
        return cls(samples=samples)


class NintendoImuEventReader:
    """Linux hid-nintendo IMU reader for Pro Controller gyro events."""

    EVENT_FORMAT = "llHHi"
    EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
    EV_ABS = 0x03
    ABS_RX = 0x03
    ABS_RY = 0x04
    ABS_RZ = 0x05
    GYRO_UNITS_PER_DPS = 14247.0

    def __init__(self, path: str | None = None) -> None:
        self.path = path or self._find_event_path()
        self._fd: int | None = None
        self._values = np.zeros(3, dtype=float)

    @staticmethod
    def _find_event_path() -> str | None:
        try:
            text = Path("/proc/bus/input/devices").read_text()
        except OSError:
            return None
        blocks = text.split("\n\n")
        for block in blocks:
            if 'Name="Nintendo Switch Pro Controller IMU"' not in block:
                continue
            for line in block.splitlines():
                if not line.startswith("H: Handlers="):
                    continue
                for token in line.removeprefix("H: Handlers=").split():
                    if token.startswith("event"):
                        return f"/dev/input/{token}"
        return None

    def connect(self) -> None:
        if self.path is None or self._fd is not None:
            return
        try:
            self._fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            self._fd = None

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    @property
    def is_available(self) -> bool:
        return self._fd is not None

    def read_gyro_dps(self) -> np.ndarray | None:
        if self._fd is None:
            return None
        while True:
            try:
                data = os.read(self._fd, self.EVENT_SIZE * 64)
            except BlockingIOError:
                break
            except OSError:
                self.close()
                return None
            if not data:
                break
            usable = len(data) // self.EVENT_SIZE * self.EVENT_SIZE
            for i in range(0, usable, self.EVENT_SIZE):
                _sec, _usec, ev_type, code, value = struct.unpack(
                    self.EVENT_FORMAT,
                    data[i : i + self.EVENT_SIZE],
                )
                if ev_type != self.EV_ABS:
                    continue
                if code == self.ABS_RX:
                    self._values[0] = value
                elif code == self.ABS_RY:
                    self._values[1] = value
                elif code == self.ABS_RZ:
                    self._values[2] = value
        return self._values / self.GYRO_UNITS_PER_DPS


class NintendoProSource:
    """The Switch Pro controller, via lerobot's `NintendoTeleop`.

    lerobot is still responsible for opening and initializing the HID device.
    The sample emitted here is deliberately closer to the raw device frame than
    lerobot's FR3-style action: `ZL` is the clutch, sticks are raw deflections,
    and the gyro-derived rotation is accumulated into the quaternion expected by
    `ClutchRetargeter`.
    """

    def __init__(self, controller: str = "pro", device_id: int | None = None) -> None:
        from lerobot.teleoperators.nintendo import NintendoTeleop, NintendoTeleopConfig
        from lerobot.teleoperators.nintendo.configuration_nintendo import NintendoController

        self._config = NintendoTeleopConfig(
            controller=NintendoController(controller),
            device_id=device_id,
            enable_rotation=True,
        )
        self._teleop = NintendoTeleop(self._config)
        self._rotation_scale = self._config.rotation_scale
        self._samples = 0
        self._imu_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self._imu_events = NintendoImuEventReader()
        self._last_imu_update_s = time.perf_counter()

    def connect(self) -> None:
        self._teleop.connect()
        self._imu_events.connect()
        self._last_imu_update_s = time.perf_counter()

    def disconnect(self) -> None:
        self._teleop.disconnect()
        self._imu_events.close()

    @property
    def is_connected(self) -> bool:
        return bool(self._teleop.is_connected)

    @staticmethod
    def _rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotation_vector))
        if angle < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rotation_vector / angle
        return np.array([np.cos(angle / 2.0), *(np.sin(angle / 2.0) * axis)])

    @staticmethod
    def _multiply_quaternions(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = np.asarray(a, float)
        bw, bx, by, bz = np.asarray(b, float)
        out = np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ]
        )
        norm = float(np.linalg.norm(out))
        return out / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])

    def _deadband(self, values: np.ndarray) -> np.ndarray:
        threshold = float(abs(self._config.stick_deadband))
        values = np.asarray(values, float)
        return np.where(np.abs(values) >= threshold, values, 0.0)

    def read(self) -> NintendoProSample:
        action = self._teleop.get_action()
        self._samples += 1
        debug = self._teleop.latest_debug_state()

        raw_left = self._deadband(np.array(debug.get("left_stick", (0.0, 0.0)), dtype=float))
        raw_right = self._deadband(np.array(debug.get("right_stick", (0.0, 0.0)), dtype=float))

        # `ClutchRetargeter` maps sample X/Y as:
        #   world_x = sample.left_stick[1], world_y = -sample.left_stick[0].
        # With the Linux Pro Controller signs this gives the desired operator
        # mapping: physical X left -> robot left, physical Y down -> robot back.
        left = raw_left

        buttons = set(debug.get("buttons", ()))
        clutch = "ZL" in buttons
        now = time.perf_counter()
        dt = min(max(now - self._last_imu_update_s, 0.0), float(self._config.max_imu_dt_s))
        self._last_imu_update_s = now

        gyro_dps = self._imu_events.read_gyro_dps()
        if gyro_dps is not None:
            gyro_dps = np.asarray(gyro_dps, float)
            if self._config.invert_wx:
                gyro_dps[0] *= -1.0
            if self._config.invert_wy:
                gyro_dps[1] *= -1.0
            if self._config.invert_wz:
                gyro_dps[2] *= -1.0
            gyro_dps = np.where(np.abs(gyro_dps) >= float(self._config.imu_gyro_deadband_dps), gyro_dps, 0.0)
            rotation_vector = np.deg2rad(gyro_dps) * dt
        else:
            rotation_vector = (
                np.array(
                    [
                        action.get("target_wx", 0.0),
                        action.get("target_wy", 0.0),
                        action.get("target_wz", 0.0),
                    ]
                )
                / max(self._rotation_scale, 1e-9)
            )

        if clutch:
            step_quaternion = self._rotation_vector_to_quaternion(rotation_vector)
            self._imu_quaternion = self._multiply_quaternions(step_quaternion, self._imu_quaternion)
        else:
            self._imu_quaternion = np.array([1.0, 0.0, 0.0, 0.0])

        return NintendoProSample(
            t=self._samples / 30.0,
            left_stick=np.clip(left, -1.0, 1.0),
            right_stick=np.array([0.0, float(np.clip(raw_right[1], -1.0, 1.0))]),
            imu_quaternion=self._imu_quaternion.copy(),
            clutch=clutch,
            gripper=float(action.get("gripper", 1.0)),
        )
