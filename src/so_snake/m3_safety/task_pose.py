"""M3a — the SO-100's five-dimensional task space.

The arm has five joints, so its end effector traces a five-dimensional manifold,
not SE(3). `docs/plan_5dof_task_space.md` fixes the coordinates on it as

    (x, y, z, pitch, roll)

and this module defines them, converts to and from rotation matrices, and
integrates operator or policy deltas into an absolute target.

## The chart

Position determines which vertical plane the arm works in. `shoulder_pan` is
the only joint that can leave that plane, and with the TCP pinned it moves by
less than +/-2.5 degrees, so the plane's azimuth is a function of position:

    psi(p) = atan2(p_y - axis_y, p_x - axis_x)

measured about the shoulder_pan axis. This is the plan's `R_position_yaw(p)`,
and it is no longer an open question: over every workspace voxel with at least
100 samples, the mean TCP yaw agrees with this closed form to a median of 0.09
degrees and a maximum of 1.02 (`scripts/build_feasibility_atlas.py`).

Given the plane, two orthogonal unit vectors span everything the arm can do to
its orientation there:

    f = (cos psi, sin psi, 0)     in-plane forward
    a = (-sin psi, cos psi, 0)    the plane's normal, and the pitch axis
    u                             the gripper's approach axis, which lies in the
                                  plane spanned by f and z

    pitch  angle of u within the plane, measured from f toward +z.
           Negative points the gripper down; -90 deg is straight down.
    roll   spin about u. Zero when the jaws close in the plane's normal
           direction.

    R_tool = Rz(psi) . Ry(-pitch) . Rx(roll)

Note the `Ry(-pitch)`: the usual Z-Y-X Euler pitch is positive *nose down*
under a Z-up world, which is a sign trap waiting to happen in a logged action
space. The coordinate is elevation, and the conversion carries the minus sign
once, here.

## Why the chart is built on position and not on the tool's own azimuth

Reading psi off the tool orientation instead -- `atan2(u_y, u_x)` -- gives
almost the same number, and a chart that is singular exactly where the gripper
points straight down. That is not an academic corner: the atlas says a third of
the workspace can reach -75 degrees and an eighth can reach -85, and pointing
straight down is what a top-down grasp of a block on a table *is*. Anchoring
psi to position instead leaves the chart regular everywhere the arm can go.

## Resolving an operator's rotation

At fixed position, `psi` cannot change, so the achievable angular velocities are
exactly

    omega  in  span{a, u}     with  a . u = 0

and because those two are orthonormal, resolving a requested rotation onto the
task coordinates is an orthogonal projection with no matrix to invert and no
configuration where it blows up:

    pitch_rate = -a . omega        roll_rate = u . omega

The discarded component lies along `a x u`, which is the direction that would
change the plane's azimuth -- the one thing the arm cannot do. Its angle from
world +Z is `pitch`, which is why the plan measured that normal wandering up to
65 degrees away from vertical: it follows the gripper, and stripping a world-Z
component instead would delete a controllable direction while leaving an
uncontrollable one in the command.

This falls out as physically correct at both ends. With the gripper horizontal,
a pure world-yaw twist is entirely rejected -- the arm has no yaw. With the
gripper pointing straight down, the same twist becomes pure roll, because at
that orientation yawing the hand and rolling the gripper are the same motion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import TaskLimits
from ..kinematics import ArmChain


def wrap_to_pi(angle: float | np.ndarray) -> np.ndarray:
    """Fold an angle into (-pi, pi]. Roll is periodic; its error must be too."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def tool_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build `R_world_tool` from the chart's angles, in radians."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(-pitch), np.sin(-pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


@dataclass(frozen=True)
class SO100TaskPose:
    """One point of the arm's real task space. Distances in m, angles in rad."""

    x: float
    y: float
    z: float
    pitch: float
    roll: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def as_array(self) -> np.ndarray:
        """`(5,)` in the frozen action-space order — the logging/policy layout."""
        return np.array([self.x, self.y, self.z, self.pitch, self.roll])

    @classmethod
    def from_array(cls, values: np.ndarray) -> SO100TaskPose:
        x, y, z, pitch, roll = np.asarray(values, float)
        return cls(float(x), float(y), float(z), float(pitch), float(roll))

    def replace(self, **changes: float) -> SO100TaskPose:
        values = {"x": self.x, "y": self.y, "z": self.z, "pitch": self.pitch, "roll": self.roll}
        values.update(changes)
        return SO100TaskPose(**values)


@dataclass(frozen=True)
class TaskFrameReadout:
    """A measured 6-DoF tool pose resolved into the chart, residual and all."""

    pose: SO100TaskPose
    yaw: float  # psi(p), from position — not read off the orientation
    yaw_residual: float  # how far the tool actually leaves the plane, rad

    @property
    def position(self) -> np.ndarray:
        return self.pose.position


class TaskFrame:
    """Converts between world tool poses and the five task coordinates."""

    def __init__(self, chain: ArmChain | None = None, pan_axis_xy: np.ndarray | None = None) -> None:
        if pan_axis_xy is None:
            pan_axis_xy = (chain or ArmChain()).pan_axis_xy()
        self.pan_axis_xy = np.asarray(pan_axis_xy, float)

    def yaw_at(self, position: np.ndarray) -> float:
        """`psi(p)` — the azimuth of the arm's working plane at a position."""
        offset = np.asarray(position, float)[:2] - self.pan_axis_xy
        return float(np.arctan2(offset[1], offset[0]))

    def plane_axes(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """In-plane forward `f` and plane normal `a` at a position."""
        yaw = self.yaw_at(position)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c, s, 0.0]), np.array([-s, c, 0.0])

    def rotation(self, pose: SO100TaskPose) -> np.ndarray:
        """`R_world_tool` for a task pose. The yaw comes from the position."""
        return tool_rotation(self.yaw_at(pose.position), pose.pitch, pose.roll)

    def to_pose_matrix(self, pose: SO100TaskPose) -> np.ndarray:
        """4x4 world tool pose. Inflating a 5D target to 6D for logging or display."""
        T = np.eye(4)
        T[:3, :3] = self.rotation(pose)
        T[:3, 3] = pose.position
        return T

    def read(self, T_world_tool: np.ndarray) -> TaskFrameReadout:
        """Resolve a measured tool pose into the chart.

        `yaw_residual` is how far the tool axis leaves the plane the chart says
        it should be in. It is the plan's `yaw_residual_for_diagnostics`, and it
        is not clamped or corrected: it is the arm telling us how good the
        `psi(p)` model is at this configuration, so silencing it would silence
        the only check on the chart itself.
        """
        T = np.asarray(T_world_tool, float)
        position = T[:3, 3]
        yaw = self.yaw_at(position)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz_T = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

        # In the plane's frame the orientation should be exactly Ry(-pitch) Rx(roll):
        #   [[ cos p, -sin p sin r, -sin p cos r],
        #    [     0,        cos r,       -sin r],
        #    [ sin p,  cos p sin r,  cos p cos r]]
        # so pitch and roll read off two entries each, and the (1, 0) entry --
        # which the model says is zero -- is the out-of-plane residual.
        R = Rz_T @ T[:3, :3]
        pitch = float(np.arctan2(R[2, 0], R[0, 0]))
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        residual = float(np.arcsin(np.clip(R[1, 0], -1.0, 1.0)))

        return TaskFrameReadout(
            pose=SO100TaskPose(float(position[0]), float(position[1]), float(position[2]), pitch, roll),
            yaw=yaw,
            yaw_residual=residual,
        )

    def rotation_axes(self, pose: SO100TaskPose) -> tuple[np.ndarray, np.ndarray]:
        """`(a, u)` — the pitch axis and the approach axis, both unit and orthogonal.

        `-a` and `u` are the plan's `b_pitch` and `b_roll`: dotting a world
        rotation with them gives the pitch and roll rates directly.
        """
        _, a = self.plane_axes(pose.position)
        u = self.rotation(pose)[:, 0]
        return a, u

    def resolve(self, omega_world: np.ndarray, pose: SO100TaskPose) -> tuple[float, float, np.ndarray]:
        """Split a world rotation into `(pitch delta, roll delta, rejected)`.

        The rejected part is returned rather than folded in anywhere: it is the
        component of the operator's gesture about an axis the arm does not have,
        and its magnitude is a diagnostic worth logging — and later a free
        feasibility label for the policy.
        """
        omega = np.asarray(omega_world, float)
        a, u = self.rotation_axes(pose)
        delta_pitch = float(-a @ omega)
        delta_roll = float(u @ omega)
        rejected = omega - (a @ omega) * a - (u @ omega) * u
        return delta_pitch, delta_roll, rejected


@dataclass
class TaskPoseUpdate:
    """What one integration step did, in the terms the loop logs and displays."""

    pose: SO100TaskPose
    position_clamped: np.ndarray  # (3,) bool, which axes hit the workspace wall
    pitch_clamped: bool
    step_rate_limited: bool

    @property
    def any_clamped(self) -> bool:
        return bool(self.position_clamped.any() or self.pitch_clamped)


@dataclass
class TaskPoseTracker:
    """Integrates operator or policy deltas into an absolute 5D target.

    Replaces `EETargetTracker`. The behavioural difference is not the
    parameterisation but what becomes impossible: the old tracker held a full
    6-DoF pose and could integrate its way into orientations the arm cannot
    hold, which is what `orientation_feasibility_feedback` existed to undo. A
    target with no yaw coordinate cannot drift in yaw, and pitch is bounded by a
    measured interval rather than an invented Euler box, so there is nothing
    left to feed back.
    """

    limits: TaskLimits = field(default_factory=TaskLimits)
    home: SO100TaskPose | None = None

    pose: SO100TaskPose = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self, pose: SO100TaskPose | None = None) -> None:
        target = pose if pose is not None else self.home
        if target is None:
            centre = self.limits.centre()
            target = SO100TaskPose(centre[0], centre[1], centre[2], 0.0, 0.0)
        self.pose = self._clamp(target)[0]

    def set_pose(self, pose: SO100TaskPose) -> None:
        """Adopt an externally measured task pose, e.g. from the arm's own FK."""
        self.reset(pose)

    def _clamp(self, pose: SO100TaskPose) -> tuple[SO100TaskPose, np.ndarray, bool]:
        position, position_clamped = self.limits.clamp_position(pose.position)
        pitch, pitch_clamped = self.limits.clamp_pitch(pose.pitch)
        clamped = SO100TaskPose(
            float(position[0]),
            float(position[1]),
            float(position[2]),
            pitch,
            float(wrap_to_pi(pose.roll)),
        )
        return clamped, position_clamped, pitch_clamped

    def approach(self, target: SO100TaskPose) -> TaskPoseUpdate:
        """Move one step toward an absolute target, through the same safety envelope.

        Teleoperation produces absolute targets (the clutch latches a pose and
        the operator moves it), while a policy produces deltas. Both arrive
        here, so the clamping and the per-step ceilings are written once and
        cannot drift apart between the two paths.
        """
        return self.update(
            target.position - self.pose.position,
            target.pitch - self.pose.pitch,
            float(wrap_to_pi(target.roll - self.pose.roll)),
        )

    def update(
        self,
        delta_position_m: np.ndarray,
        delta_pitch_rad: float = 0.0,
        delta_roll_rad: float = 0.0,
    ) -> TaskPoseUpdate:
        """Advance the target by one control step.

        Deltas arrive already in metres and radians -- the normalised-input
        scaling belongs to whoever produced them, so that a policy action and an
        operator action reach this method in identical units.
        """
        step = np.asarray(delta_position_m, float).copy()
        rate_limited = False

        norm = float(np.linalg.norm(step))
        if norm > self.limits.max_step_pos_m:
            step *= self.limits.max_step_pos_m / norm
            rate_limited = True

        rotation_step = np.array([float(delta_pitch_rad), float(delta_roll_rad)])
        rotation_norm = float(np.linalg.norm(rotation_step))
        if rotation_norm > self.limits.max_step_rot_rad:
            rotation_step *= self.limits.max_step_rot_rad / rotation_norm
            rate_limited = True

        proposed = SO100TaskPose(
            self.pose.x + step[0],
            self.pose.y + step[1],
            self.pose.z + step[2],
            self.pose.pitch + rotation_step[0],
            self.pose.roll + rotation_step[1],
        )
        self.pose, position_clamped, pitch_clamped = self._clamp(proposed)
        return TaskPoseUpdate(
            pose=self.pose,
            position_clamped=position_clamped,
            pitch_clamped=pitch_clamped,
            step_rate_limited=rate_limited,
        )
