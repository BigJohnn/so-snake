"""Forward kinematics and the geometric Jacobian, straight from the URDF.

The 5D task solver needs `J(q)` every control step, and it needs it to be exact.
placo gives forward kinematics but no Jacobian we can reach, and differencing FK
numerically costs five extra solver calls per step and introduces an error floor
right where the damping schedule has to read the smallest singular value.

The SO-100's arm is a six-transform serial chain, so the analytic Jacobian is a
dozen lines: for a revolute joint,

    J_v[:, i] = z_i x (p_tcp - p_i)      J_w[:, i] = z_i

with `z_i` the joint axis and `p_i` the joint origin, both in the frame the
Jacobian is expressed in. This module is therefore the control path's kinematics,
with placo and MuJoCo kept as independent implementations to check it against --
see `scripts/check_kinematics_agreement.py`, which pins all three to 1e-9 m.

Everything here is in the *world* frame (the +X-forward convention of
`ArmConfig.world_from_base_yaw_deg`), not the URDF base frame, because every
consumer -- workspace box, operator intuition, task pose -- works in world.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GRIPPER_JOINT, ArmConfig


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF's fixed-axis roll-pitch-yaw convention: R = Rz(y) Ry(p) Rx(r)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula for a rotation of `angle` about a unit `axis`."""
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """`(w, x, y, z)` to a rotation matrix. Normalises first, so drift is harmless."""
    q = np.asarray(quaternion, float)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("cannot build a rotation from a zero quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_log(R: np.ndarray) -> np.ndarray:
    """`log(R)` as an axis-angle vector: the rotation's axis scaled by its angle.

    Handles both ends of the range, which the naive formula does not: near zero
    the axis is recovered from the antisymmetric part before dividing by a
    vanishing sine, and near pi it is recovered from `R + I`, whose columns are
    all parallel to the axis.
    """
    R = np.asarray(R, float)
    cosine = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    antisymmetric = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if angle < 1e-8:
        return 0.5 * antisymmetric
    if angle < np.pi - 1e-6:
        return angle * antisymmetric / (2.0 * np.sin(angle))

    symmetric = R + np.eye(3)
    axis = symmetric[:, int(np.argmax(np.linalg.norm(symmetric, axis=0)))]
    axis = axis / np.linalg.norm(axis)
    if antisymmetric @ axis < 0:
        axis = -axis
    return angle * axis


@dataclass(frozen=True)
class UrdfJoint:
    """One joint's fixed placement plus, if it moves, its axis."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray  # 4x4, child frame relative to parent at zero
    axis: np.ndarray  # (3,) unit, in the child frame; zeros for fixed joints

    @property
    def is_revolute(self) -> bool:
        return self.joint_type in ("revolute", "continuous")


def _parse_joints(urdf_path: Path) -> dict[str, UrdfJoint]:
    joints: dict[str, UrdfJoint] = {}
    for element in ET.parse(urdf_path).getroot().findall("joint"):
        origin = np.eye(4)
        origin_el = element.find("origin")
        if origin_el is not None:
            origin[:3, 3] = np.fromstring(origin_el.get("xyz", "0 0 0"), sep=" ")
            origin[:3, :3] = _rpy_to_matrix(np.fromstring(origin_el.get("rpy", "0 0 0"), sep=" "))

        axis = np.zeros(3)
        axis_el = element.find("axis")
        if axis_el is not None:
            raw = np.fromstring(axis_el.get("xyz", "0 0 0"), sep=" ")
            norm = np.linalg.norm(raw)
            axis = raw / norm if norm > 0 else raw

        name = element.get("name", "")
        joints[name] = UrdfJoint(
            name=name,
            joint_type=element.get("type", "fixed"),
            parent=element.find("parent").get("link", ""),
            child=element.find("child").get("link", ""),
            origin=origin,
            axis=axis,
        )
    return joints


class ArmChain:
    """The base-to-TCP serial chain, in the world frame.

    Angles are in degrees throughout the public interface, matching the servo
    units the rest of the system speaks. Internals work in radians.
    """

    def __init__(self, arm: ArmConfig | None = None) -> None:
        self.arm = arm or ArmConfig()
        if not self.arm.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.arm.urdf_path}")

        self._joints = _parse_joints(self.arm.urdf_path)
        child_of = {j.child: j for j in self._joints.values()}

        # Walk up from the TCP to the root, then reverse: this picks out the
        # chain that actually reaches the tool and silently drops branches such
        # as the gripper jaw.
        chain: list[UrdfJoint] = []
        link = self.arm.tcp_frame
        while link in child_of:
            joint = child_of[link]
            chain.append(joint)
            link = joint.parent
        self._chain = list(reversed(chain))
        self._root_link = link

        actuated = [j.name for j in self._chain if j.is_revolute]
        if actuated != list(self.arm.joint_names):
            raise ValueError(
                f"URDF chain to {self.arm.tcp_frame!r} actuates {actuated}, "
                f"but ArmConfig.joint_names is {list(self.arm.joint_names)}"
            )

        self._world_from_base = self.arm.world_from_base()
        self.n_joints = len(actuated)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.arm.joint_names

    @property
    def root_link(self) -> str:
        return self._root_link

    def _frames(self, q_rad: np.ndarray) -> list[np.ndarray]:
        """Cumulative world-frame transforms, one entry per joint in the chain.

        Entry `i` is the pose of joint `i`'s child link. The last entry is the
        TCP.
        """
        out: list[np.ndarray] = []
        T = self._world_from_base
        k = 0
        for joint in self._chain:
            T = T @ joint.origin
            if joint.is_revolute:
                R = np.eye(4)
                R[:3, :3] = axis_angle_to_matrix(joint.axis, float(q_rad[k]))
                T = T @ R
                k += 1
            out.append(T)
        return out

    def fk(self, joints_deg: np.ndarray) -> np.ndarray:
        """TCP pose in the world frame, 4x4."""
        q = np.deg2rad(np.asarray(joints_deg, float)[: self.n_joints])
        return self._frames(q)[-1]

    def _frames_batch(self, q_rad: np.ndarray) -> list[np.ndarray]:
        """:meth:`_frames`, hoisted over a leading batch dimension."""
        n = q_rad.shape[0]
        out: list[np.ndarray] = []
        T = np.broadcast_to(self._world_from_base, (n, 4, 4)).copy()
        k = 0
        for joint in self._chain:
            T = T @ joint.origin
            if joint.is_revolute:
                angle = q_rad[:, k]
                K = np.array(
                    [
                        [0.0, -joint.axis[2], joint.axis[1]],
                        [joint.axis[2], 0.0, -joint.axis[0]],
                        [-joint.axis[1], joint.axis[0], 0.0],
                    ]
                )
                R = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
                R[:, :3, :3] = (
                    np.eye(3)
                    + np.sin(angle)[:, None, None] * K
                    + (1.0 - np.cos(angle))[:, None, None] * (K @ K)
                )
                T = T @ R
                k += 1
            out.append(T)
        return out

    def fk_batch(self, joints_deg: np.ndarray) -> np.ndarray:
        """TCP poses for `(N, 5)` configurations at once, `(N, 4, 4)`, world frame.

        Same arithmetic as :meth:`fk`, hoisted over the batch dimension. The
        feasibility atlas forward-solves millions of configurations, which is
        minutes in a Python loop and seconds here.
        """
        q = np.deg2rad(np.atleast_2d(np.asarray(joints_deg, float))[:, : self.n_joints])
        return self._frames_batch(q)[-1]

    def jacobian_batch(self, joints_deg: np.ndarray) -> np.ndarray:
        """Geometric Jacobians for `(N, 5)` configurations, `(N, 6, 5)`, world frame."""
        q = np.deg2rad(np.atleast_2d(np.asarray(joints_deg, float))[:, : self.n_joints])
        frames = self._frames_batch(q)
        p_tcp = frames[-1][:, :3, 3]

        J = np.zeros((q.shape[0], 6, self.n_joints))
        k = 0
        for joint, T in zip(self._chain, frames, strict=True):
            if not joint.is_revolute:
                continue
            z = T[:, :3, :3] @ joint.axis
            J[:, :3, k] = np.cross(z, p_tcp - T[:, :3, 3])
            J[:, 3:, k] = z
            k += 1
        return J

    def fk_all(self, joints_deg: np.ndarray) -> dict[str, np.ndarray]:
        """World pose of every link along the chain, keyed by link name."""
        q = np.deg2rad(np.asarray(joints_deg, float)[: self.n_joints])
        frames = self._frames(q)
        return {joint.child: T for joint, T in zip(self._chain, frames, strict=True)}

    def jacobian(self, joints_deg: np.ndarray) -> np.ndarray:
        """Geometric Jacobian at the TCP, 6x5, world frame.

        Rows 0-2 map joint velocity (rad/s) to TCP linear velocity (m/s), rows
        3-5 to angular velocity (rad/s) about world axes.
        """
        q = np.deg2rad(np.asarray(joints_deg, float)[: self.n_joints])
        frames = self._frames(q)
        p_tcp = frames[-1][:3, 3]

        J = np.zeros((6, self.n_joints))
        k = 0
        for joint, T in zip(self._chain, frames, strict=True):
            if not joint.is_revolute:
                continue
            z = T[:3, :3] @ joint.axis
            J[:3, k] = np.cross(z, p_tcp - T[:3, 3])
            J[3:, k] = z
            k += 1
        return J

    def pan_axis_xy(self) -> np.ndarray:
        """Where the shoulder_pan axis crosses the world XY plane.

        The task-space chart measures azimuth about this point, not about the
        world origin. They differ by 45 mm, which is 10 degrees of azimuth at
        the near edge of the workspace -- not a rounding error.
        """
        origin = self._world_from_base @ self._joints[self.arm.joint_names[0]].origin
        return origin[:2, 3].copy()

    def joint_axis_world(self, joint_name: str, joints_deg: np.ndarray) -> np.ndarray:
        """Unit direction of one joint's axis in the world frame."""
        q = np.deg2rad(np.asarray(joints_deg, float)[: self.n_joints])
        for joint, T in zip(self._chain, self._frames(q), strict=True):
            if joint.name == joint_name:
                return T[:3, :3] @ joint.axis
        raise KeyError(f"{joint_name!r} is not on the chain to {self.arm.tcp_frame!r}")

    def tool_from_tcp(self) -> np.ndarray:
        """Constant 3x3 rotation from the URDF TCP frame to the tool convention.

        Task-space pitch and roll are meaningless until "which way does the tool
        point" and "roll about what" are pinned to physical features of the
        gripper, so this frame is derived from the mechanism rather than chosen:

            +X  the wrist_roll axis, pointing out of the wrist toward the TCP.
                The one direction the arm can spin freely, so it is what "roll"
                rolls about and what "pitch" tilts.
            +Z  the gripper jaw's hinge axis, perpendicular to +X by
                construction of the wrist.
            +Y  Z x X, which is the direction the jaws close along.

        The sign of +X is not free: `wrist_roll`'s URDF axis points back into
        the wrist, so it is flipped to point along the TCP's offset from the
        joint. Without that flip the tool appears to face backwards and every
        pitch reads with the wrong sign.

        Returns `R_tcp_tool`, whose columns are the tool axes in TCP
        coordinates. `R_world_tool = fk(q)[:3, :3] @ tool_from_tcp()`.
        """
        tcp_joint = self._joints[self._chain[-1].name]
        roll_joint = self._joints["wrist_roll"]
        R_link_tcp = tcp_joint.origin[:3, :3]

        # Both axes live in the frame of the link the TCP hangs off, so they
        # transfer into TCP coordinates through that one fixed rotation.
        roll_axis = roll_joint.axis
        if float(tcp_joint.origin[:3, 3] @ roll_axis) < 0.0:
            roll_axis = -roll_axis
        x_tool = R_link_tcp.T @ roll_axis

        jaw_joint = self._joints.get(GRIPPER_JOINT)
        if jaw_joint is None:
            raise KeyError(f"URDF has no {GRIPPER_JOINT!r} joint to take the jaw hinge axis from")
        z_tool = R_link_tcp.T @ (jaw_joint.origin[:3, :3] @ jaw_joint.axis)

        # Re-orthogonalise: the URDF writes right angles as 1.57079, so the two
        # axes are a few microradians off perpendicular.
        x_tool = x_tool / np.linalg.norm(x_tool)
        z_tool = z_tool - (z_tool @ x_tool) * x_tool
        z_tool = z_tool / np.linalg.norm(z_tool)
        y_tool = np.cross(z_tool, x_tool)
        return np.column_stack([x_tool, y_tool, z_tool])
