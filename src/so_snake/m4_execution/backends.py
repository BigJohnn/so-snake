"""M4 — the execution module: whatever actually receives joint commands.

Two backends behind one interface, so the control loop above is identical
offline and on hardware:

  MockFollower       servo model with first-order lag and hard limits, no hardware
  SOFollowerBackend  lerobot's Feetech STS3215 driver on the real SO-100

The mock is not a physics simulation and does not pretend to be. It exists to
exercise the parts of the pipeline that break for boring reasons -- wrong joint
order, wrong units, a stale observation feeding back into IK, a control loop
that silently runs at 4 Hz -- which is where Phase 0 time actually goes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import ArmConfig, GRIPPER_JOINT, TeleopConfig
from .joint_map import JointFrameMap


@runtime_checkable
class RobotBackend(Protocol):
    """Minimal joint-space interface the teleoperation loop depends on."""

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Arm joints followed by the gripper."""
        ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    @property
    def is_connected(self) -> bool: ...

    def read_joints_deg(self) -> np.ndarray:
        """Measured positions, in `joint_names` order."""
        ...

    def write_joints_deg(self, target_deg: np.ndarray) -> None:
        """Command positions, in `joint_names` order."""
        ...


@dataclass
class MockFollower:
    """A servo model good enough to catch integration bugs, and nothing more.

    Each joint tracks its target as a first-order lag, which reproduces the one
    property of real servos that most often breaks a naive control loop: the
    measurement fed back into IK is *behind* the command, so a loop that seeds
    IK from the measurement and assumes it arrived converges differently than
    one that seeds from the last command.
    """

    arm: ArmConfig = field(default_factory=ArmConfig)

    # Fraction of the remaining error covered per control step. 1.0 is a
    # perfect instantaneous servo; the STS3215 under a light load at 30 Hz is
    # nearer 0.35, which is the default here.
    tracking_gain: float = 0.35
    # Standard deviation of the position read-back noise, in degrees. The
    # STS3215's 12-bit encoder over its 360 deg range quantises at ~0.088 deg.
    read_noise_deg: float = 0.05
    initial_joints_deg: np.ndarray | None = None
    seed: int = 0

    _position: np.ndarray = field(init=False)
    _connected: bool = field(default=False, init=False)
    _rng: np.random.Generator = field(init=False)
    write_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        if self.initial_joints_deg is not None:
            self._position = np.asarray(self.initial_joints_deg, float).copy()
        else:
            # Start at the teleoperation home configuration. Mid-range joints
            # would seem the neutral choice but put the TCP 65 mm outside the
            # workspace box, so every run began with the arm travelling back in
            # and a large transient tracking error that looked like an IK fault.
            teleop = TeleopConfig()
            gripper_mid = sum(self.arm.joint_limits_deg[GRIPPER_JOINT]) / 2.0
            self._position = np.array([*teleop.home_joints_deg, gripper_mid], dtype=float)
        self._lo = np.array([self.arm.joint_limits_deg[j][0] for j in self.joint_names])
        self._hi = np.array([self.arm.joint_limits_deg[j][1] for j in self.joint_names])

    @property
    def joint_names(self) -> tuple[str, ...]:
        return (*self.arm.joint_names, GRIPPER_JOINT)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("MockFollower is not connected; call connect() first")

    def read_joints_deg(self) -> np.ndarray:
        self._require_connection()
        noise = self._rng.normal(0.0, self.read_noise_deg, size=self._position.shape)
        return np.clip(self._position + noise, self._lo, self._hi)

    def write_joints_deg(self, target_deg: np.ndarray) -> None:
        self._require_connection()
        target = np.clip(np.asarray(target_deg, float), self._lo, self._hi)
        if target.shape != self._position.shape:
            raise ValueError(
                f"expected {self._position.shape[0]} joint values in order {self.joint_names}, "
                f"got shape {target.shape}"
            )
        self._position = self._position + self.tracking_gain * (target - self._position)
        self.write_count += 1

    def true_joints_deg(self) -> np.ndarray:
        """Noise-free state. Available on the mock only, for test assertions."""
        return self._position.copy()


class SOFollowerBackend:
    """The real SO-100, via lerobot's `SOFollower`.

    Config path validated against the lab's lerobot (`linkage-x` `box` branch);
    not yet exercised against a moving arm. Before this is trusted the port must
    be identified and the motors calibrated on the physical robot -- run
    `scripts/preflight_real_arm.py`, which checks both without moving anything.

    `max_relative_target` is lerobot's per-step safety clamp: the magnitude of
    the relative move requested in one `send_action` is clipped to this (a scalar
    for all motors, or a per-motor dict). It is a hardware-level backstop behind
    the loop's own `max_joint_step_deg`; leave it `None` only once teleop is
    trusted.

    `joint_map` reconciles lerobot's calibration frame with the URDF frame the
    control loop works in: reads are mapped lerobot->URDF and writes URDF->lerobot
    (see `joint_map.py`). Without it the loop would drive the arm in the wrong
    frame; it must be supplied for real teleop. `None` passes values through raw,
    for bring-up diagnostics only.
    """

    def __init__(
        self,
        port: str,
        arm: ArmConfig | None = None,
        robot_id: str = "so_snake",
        *,
        max_relative_target: float | dict[str, float] | None = None,
        joint_map: JointFrameMap | None = None,
    ) -> None:
        # SOFollowerRobotConfig = RobotConfig + SOFollowerConfig. The bare
        # SOFollowerConfig has no `id`/`calibration_dir`, so lerobot's base
        # Robot.__init__ (which reads `config.id`) raises on it. The registered
        # subclass carries both, plus the max_relative_target safety clamp.
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        self.arm = arm or ArmConfig()
        self.joint_map = joint_map
        self._config = SOFollowerRobotConfig(
            port=port,
            id=robot_id,
            max_relative_target=max_relative_target,
            use_degrees=True,
        )
        self._robot = SOFollower(self._config)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return (*self.arm.joint_names, GRIPPER_JOINT)

    @property
    def max_relative_target(self) -> float | dict[str, float] | None:
        return self._config.max_relative_target

    def set_max_relative_target(self, clamp: float | dict[str, float] | None) -> None:
        """Change the per-step hardware clamp on a connected arm.

        lerobot reads `config.max_relative_target` inside every `send_action`,
        so this takes effect on the next command with no reconnect. That matters
        because the alternative -- rebuilding the backend to change one number --
        means a disconnect, and a disconnect drops torque and lets the arm sag.
        """
        self._config.max_relative_target = clamp

    def connect(self) -> None:
        self._robot.connect()

    def disconnect(self) -> None:
        # Drop torque on every motor while the port is still fully open, THEN do
        # the normal disconnect. lerobot's own disable_torque_on_disconnect clears
        # the port before issuing the disable writes, which can leave the last
        # motor on the bus -- the gripper -- still energized. Doing it explicitly
        # here first guarantees the gripper relaxes when the program exits.
        try:
            if self._robot.is_connected:
                self._robot.bus.disable_torque(num_retry=5)
        except Exception:  # noqa: BLE001 - best effort; the normal path runs next
            pass
        self._robot.disconnect()

    @property
    def is_connected(self) -> bool:
        return bool(self._robot.is_connected)

    def read_joints_deg(self) -> np.ndarray:
        """Measured joints in URDF degrees (mapped from lerobot's frame)."""
        obs = self._robot.get_observation()
        lerobot = [obs[f"{name}.pos"] for name in self.joint_names]
        if self.joint_map is None:
            return np.array(lerobot, dtype=float)
        return np.array(
            [self.joint_map.lerobot_to_urdf(name, v) for name, v in zip(self.joint_names, lerobot)],
            dtype=float,
        )

    def write_joints_deg(self, target_deg: np.ndarray) -> None:
        """Command joints given in URDF degrees (mapped back into lerobot's frame)."""
        target = np.asarray(target_deg, float)
        if self.joint_map is None:
            values = target
        else:
            values = [self.joint_map.urdf_to_lerobot(name, v) for name, v in zip(self.joint_names, target)]
        action = {f"{name}.pos": float(v) for name, v in zip(self.joint_names, values, strict=True)}
        self._robot.send_action(action)
