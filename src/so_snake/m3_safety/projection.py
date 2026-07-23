"""M3 — the orientation projector (plan step 1).

Answers one question per control step: the operator turned their hand by some
world-frame rotation, so what should the 5D target's pitch and roll do?

The chart in `task_pose` already gives the answer analytically -- at fixed
position the achievable rotations are exactly `span{a, u}` for the plane normal
`a` and approach axis `u`, so the resolution is an orthogonal projection onto
two vectors. This module wraps that, and cross-checks it against the definition
the plan works from,

    B = J_w . null(J_v)

the angular directions actually available with the TCP pinned, measured from
the Jacobian at the current configuration and cached nowhere. The two agree
because they are two descriptions of the same plane, and `basis_disagreement`
reports by how much, in degrees, so that the agreement is a measurement rather
than an assumption. `scripts/check_projection.py` runs it over the workspace.

Keeping the Jacobian route alive is not ceremony. `span{a, u}` is derived from a
model of the arm -- that `shoulder_pan` alone sets the working plane -- and if
hardware calibration ever breaks that model, the Jacobian version notices and
the analytic one does not.

Two further things fall out of `B` and are logged rather than acted on:

  * `rejected_rotation_norm` — how much of the operator's gesture was about the
    one axis the arm has no joint for. Large values mean they are pushing
    against a limit of the mechanism, which is also a free feasibility label for
    the active-perception policy later.
  * the two singular values of `B`, which differ by 2-3x. The same commanded
    rotation costs two to three times more joint motion in one direction than
    the other, and the IK's damping schedule reads the smaller one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..kinematics import ArmChain
from .task_pose import SO100TaskPose, TaskFrame


@dataclass
class ControllableBasis:
    """The orientation directions available at one configuration, TCP pinned."""

    basis: np.ndarray  # (3, 2) orthonormal columns spanning the achievable rotations
    normal: np.ndarray  # (3,) unit normal to that plane — the direction the arm lacks
    singular_values: np.ndarray  # (2,) gains of J_w . null(J_v), largest first
    rank: int  # numerical rank of J_w . null(J_v); the arm's claim is that this is 2

    @property
    def gain_ratio(self) -> float:
        """How much stiffer the weak direction is than the strong one."""
        weak = float(self.singular_values[1])
        return float(self.singular_values[0]) / weak if weak > 1e-12 else float("inf")


@dataclass
class ProjectedRotation:
    """One operator rotation resolved into the task space, with what was left over."""

    delta_pitch: float  # rad
    delta_roll: float  # rad
    omega_rejected: np.ndarray  # (3,) part of the request the arm cannot perform
    basis: ControllableBasis | None  # None unless the Jacobian check was requested

    @property
    def rejected_norm(self) -> float:
        return float(np.linalg.norm(self.omega_rejected))


class OrientationProjector:
    """Resolves world-frame rotation requests into `(pitch, roll)` deltas."""

    # Singular values of `B` run around 0.006-0.018, so a rank threshold has to
    # be scaled off the largest one rather than being absolute. 1e-6 relative is
    # far below the measured second singular value and far above the third,
    # which comes out at machine zero.
    rank_rtol: float = 1e-6

    def __init__(self, chain: ArmChain | None = None, frame: TaskFrame | None = None) -> None:
        self.chain = chain or ArmChain()
        self.frame = frame or TaskFrame(self.chain)
        self._R_tcp_tool = self.chain.tool_from_tcp()

    def tool_pose(self, joints_deg: np.ndarray) -> np.ndarray:
        """`T_world_tool` at this configuration, 4x4."""
        T = self.chain.fk(joints_deg)
        T[:3, :3] = T[:3, :3] @ self._R_tcp_tool
        return T

    def controllable_basis(self, joints_deg: np.ndarray) -> ControllableBasis:
        """`B = J_w . null(J_v)`, orthonormalised, plus the direction it misses.

        Recomputed from scratch every call. Caching it across configurations is
        the specific bug this design exists to prevent: the normal to `span(B)`
        follows the gripper's pitch, so a stale basis silently projects onto the
        wrong plane.
        """
        J = self.chain.jacobian(joints_deg)
        J_v, J_w = J[:3], J[3:]

        # Right singular vectors of J_v beyond its rank span the null space.
        _, s_v, Vt = np.linalg.svd(J_v)
        rank_v = int((s_v > s_v[0] * self.rank_rtol).sum()) if s_v.size else 0
        null_space = Vt[rank_v:].T  # (5, 5 - rank_v)

        U, s, _ = np.linalg.svd(J_w @ null_space)
        rank = int((s > s[0] * self.rank_rtol).sum()) if s.size and s[0] > 0 else 0

        return ControllableBasis(
            basis=U[:, :2],
            normal=U[:, 2],
            singular_values=np.array([s[0] if s.size > 0 else 0.0, s[1] if s.size > 1 else 0.0]),
            rank=rank,
        )

    def basis_disagreement_deg(self, joints_deg: np.ndarray) -> float:
        """Angle between the Jacobian's controllable plane and the chart's, in degrees.

        Zero would mean `span{a, u}` is exactly `span(J_w . null(J_v))`. What
        this actually measures is how well "shoulder_pan alone sets the working
        plane" describes the real chain -- the same modelling assumption the
        `psi(p)` closed form rests on, checked from the other direction.
        """
        readout = self.frame.read(self.tool_pose(joints_deg))
        a, u = self.frame.rotation_axes(readout.pose)
        normal = self.controllable_basis(joints_deg).normal
        analytic_normal = np.cross(a, u)
        analytic_normal /= np.linalg.norm(analytic_normal)
        alignment = abs(float(normal @ analytic_normal))
        return float(np.degrees(np.arccos(np.clip(alignment, -1.0, 1.0))))

    def project(
        self,
        omega_world: np.ndarray,
        pose: SO100TaskPose,
        *,
        joints_deg: np.ndarray | None = None,
    ) -> ProjectedRotation:
        """Resolve a world-frame rotation request at a task pose.

        Args:
            omega_world: Requested rotation as a world-frame axis-angle vector
                (or angular velocity — the map is linear, so units come out as
                they went in).
            pose: The task pose the request is being made from. The chart needs
                only this, not the joint configuration.
            joints_deg: Supply to also compute the Jacobian's view of what is
                controllable here, at the cost of one Jacobian evaluation.
                Diagnostics only; the projection itself does not use it.
        """
        delta_pitch, delta_roll, rejected = self.frame.resolve(omega_world, pose)
        basis = None if joints_deg is None else self.controllable_basis(joints_deg)
        return ProjectedRotation(
            delta_pitch=delta_pitch,
            delta_roll=delta_roll,
            omega_rejected=rejected,
            basis=basis,
        )
