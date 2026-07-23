"""M3b — inverse kinematics for the arm's real task space (plan step 2).

Five joints, five task coordinates, one square system:

    e(q)   = [ p* - p(q) ,  pitch* - pitch(q) ,  wrap(roll* - roll(q)) ]   in R^5
    J_task = [ J_v ; b_pitch^T J_w ; b_roll^T J_w ]                        in R^{5x5}
    dq     = J_task^T (J_task J_task^T + lambda^2 I)^{-1} e

There is no orientation weight to tune and no yaw objective to compromise
against, because the target has no yaw. That is the whole point of the
migration: `ik_orientation_weight` existed only to arbitrate between a 6-DoF
request and a 5-DoF arm, and with the request written honestly the arbitration
disappears.

Replacing placo's frame task also removes three measured problems, none of them
about degrees of freedom (see `docs/plan_5dof_task_space.md`): its solver
carries state between calls so the same target solves differently depending on
history; it fell into local minima on 17 points of a 14^3 workspace grid, the
worst missing by 360 mm; and its multi-seed workaround risked jumping IK
branches mid-motion. This solver is stateless, and a teleoperation loop always
seeds it from the previous step, which is what a local method wants.

placo is kept as an independent forward-kinematics implementation to check
against, not in the control path. So is MuJoCo -- see `so_snake.sim`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ArmConfig, IK5DConfig, TeleopConfig
from ..kinematics import ArmChain
from .task_pose import SO100TaskPose, TaskFrame, wrap_to_pi


@dataclass
class IK5DResult:
    """Outcome of one solve, including everything the safety envelope changed."""

    joints_deg: np.ndarray  # (5,) commanded joints, post-clamping
    raw_joints_deg: np.ndarray  # (5,) as the solver converged, pre-rate-limit
    achieved: SO100TaskPose  # what `joints_deg` actually reaches
    achieved_yaw_rad: float  # psi(p) at the achieved position
    achieved_yaw_residual_rad: float  # how far the tool left the plane psi(p) predicts
    achieved_pose_world: np.ndarray  # 4x4 tool pose, for logging the full 6-DoF state
    position_error_m: float
    pitch_error_rad: float
    roll_error_rad: float
    iterations: int
    converged: bool  # post-limit command reaches the target within tolerance
    solver_converged: bool  # raw DLS iteration reached the target before joint safety
    min_singular_value: float  # of J_task at the solution — closeness to singularity
    damping: float  # lambda actually used on the last iteration
    limit_clamped: np.ndarray  # (5,) bool
    rate_clamped: np.ndarray  # (5,) bool

    @property
    def any_clamped(self) -> bool:
        return bool(self.limit_clamped.any() or self.rate_clamped.any())

    @property
    def orientation_error_rad(self) -> float:
        return float(np.hypot(self.pitch_error_rad, self.roll_error_rad))


class TaskIK5D:
    """Damped least squares in `(x, y, z, pitch, roll)`, with joint safety applied."""

    def __init__(
        self,
        chain: ArmChain | None = None,
        arm: ArmConfig | None = None,
        teleop: TeleopConfig | None = None,
        ik: IK5DConfig | None = None,
        frame: TaskFrame | None = None,
    ) -> None:
        self.arm = arm or ArmConfig()
        self.chain = chain or ArmChain(self.arm)
        self.teleop = teleop or TeleopConfig()
        self.cfg = ik or IK5DConfig()
        self.frame = frame or TaskFrame(self.chain)
        self._lo, self._hi = self.arm.limits_deg_array()
        self._lo_rad, self._hi_rad = np.deg2rad(self._lo), np.deg2rad(self._hi)
        self._R_tcp_tool = self.chain.tool_from_tcp()
        self._n = self.chain.n_joints

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.arm.joint_names

    def forward(self, joints_deg: np.ndarray) -> np.ndarray:
        """Tool pose in the world frame, 4x4."""
        T = self.chain.fk(joints_deg)
        T[:3, :3] = T[:3, :3] @ self._R_tcp_tool
        return T

    def task_pose(self, joints_deg: np.ndarray):
        """Measured task pose, plus the plane azimuth and out-of-plane residual."""
        return self.frame.read(self.forward(joints_deg))

    def seed_for(self, target: SO100TaskPose) -> np.ndarray:
        """A cold-start seed built from the target, for callers with no history.

        Two of the five joints can be read almost directly off a 5D target:
        `shoulder_pan` is the plane azimuth `psi(p)`, to within 0.06 deg, and
        `wrist_roll` is `90 deg - roll`, since roll is defined about the wrist
        axis. The remaining three set the arm's posture within the plane and are
        taken from the home configuration.

        This is not a substitute for seeding from the previous command. In
        closed loop, always pass the previous command: a solver reseeded from a
        target-derived guess every step is free to change IK branch between
        steps, which is a lurch.
        """
        seed = np.array(self.teleop.home_joints_deg, dtype=float)
        seed[0] = np.degrees(self.frame.yaw_at(target.position))
        seed[4] = float((90.0 - np.degrees(target.roll) + 180.0) % 360.0 - 180.0)
        return np.clip(seed, self._lo, self._hi)

    def _task_error_and_jacobian(
        self, q_rad: np.ndarray, target: SO100TaskPose
    ) -> tuple[np.ndarray, np.ndarray]:
        joints_deg = np.rad2deg(q_rad)
        T = self.chain.fk(joints_deg)
        T[:3, :3] = T[:3, :3] @ self._R_tcp_tool
        readout = self.frame.read(T)

        error = np.empty(5)
        error[:3] = target.position - readout.position
        error[3] = target.pitch - readout.pose.pitch
        error[4] = float(wrap_to_pi(target.roll - readout.pose.roll))

        # The task rows are the plan's `b_pitch^T J_w` and `b_roll^T J_w`. Both
        # axes are taken at the *current* configuration, not the target's: the
        # Jacobian has to describe the arm where it presently is, or the step
        # points somewhere other than downhill.
        a, u = self.frame.rotation_axes(readout.pose)
        J = self.chain.jacobian(joints_deg)
        J_task = np.vstack([J[:3], -a @ J[3:], u @ J[3:]])
        return error, J_task

    def _damping(self, singular_values: np.ndarray) -> float:
        """Levenberg-style schedule: damp only when the system is going singular.

        A fixed lambda is a permanent accuracy tax paid everywhere in order to
        survive the small part of the workspace that needs it. Below
        `singular_threshold` the damping ramps up quadratically, which keeps
        `dq` bounded through the singularity instead of letting the pseudo-
        inverse blow up; above it, damping stays at the floor and the solve is
        effectively exact.
        """
        s_min = float(singular_values[-1])
        if s_min >= self.cfg.singular_threshold:
            return self.cfg.damping_min
        ratio = s_min / self.cfg.singular_threshold
        return float(
            np.hypot(self.cfg.damping_min, self.cfg.damping_max * (1.0 - ratio * ratio))
        )

    def solve(
        self,
        seed_joints_deg: np.ndarray,
        target: SO100TaskPose,
        *,
        rate_limit: bool = True,
        rate_reference_deg: np.ndarray | None = None,
    ) -> IK5DResult:
        """Solve for joints reaching `target`, starting from `seed_joints_deg`.

        Args:
            seed_joints_deg: Where the iteration starts. In closed loop this
                should be the previous *command*, not the measurement: the servo
                lags, so seeding from where the arm currently is drags the
                solution backwards and lets successive solves drift between IK
                branches.
            target: The 5D goal. Yaw is not part of it and is not solved for.
            rate_limit: Apply the per-step joint velocity ceiling. Disable only
                for offline analysis, where step size means nothing.
            rate_reference_deg: What the velocity cap is measured against.
                Defaults to `seed_joints_deg`.
        """
        seed = np.asarray(seed_joints_deg, float)[: self._n]
        q = np.clip(np.deg2rad(seed), self._lo_rad, self._hi_rad)

        damping = self.cfg.damping_min
        s_min = float("nan")
        converged = False
        iterations = 0
        previous_residual = np.inf
        weights = np.array([1.0, 1.0, 1.0, self.cfg.angle_weight_m, self.cfg.angle_weight_m])

        for iterations in range(1, self.cfg.max_iterations + 1):
            error, J_task = self._task_error_and_jacobian(q, target)

            if (
                np.linalg.norm(error[:3]) <= self.cfg.position_tolerance_m
                and abs(error[3]) <= self.cfg.angle_tolerance_rad
                and abs(error[4]) <= self.cfg.angle_tolerance_rad
            ):
                converged = True
                break

            # Weight the two angle rows into length units so that near a
            # singularity, where the damping forces a compromise, it is spent on
            # orientation rather than on position -- a grasp survives a degree of
            # approach-angle error far better than a millimetre of placement.
            e_weighted = error * weights
            residual = float(np.linalg.norm(e_weighted))

            # Heavy damping near a singularity shrinks the step along the weak
            # direction by orders of magnitude, so the iteration can spend its
            # whole budget crawling the last few micrometres of a residual that
            # is already far inside grasp tolerance. Stop when progress stalls
            # and report the residual honestly rather than burning the budget.
            if residual > previous_residual * (1.0 - self.cfg.min_relative_progress):
                break
            previous_residual = residual

            J_weighted = J_task * weights[:, None]
            s = np.linalg.svd(J_weighted, compute_uv=False)
            s_min = float(s[-1])
            damping = self._damping(s)

            A = J_weighted @ J_weighted.T + (damping**2) * np.eye(5)
            dq = J_weighted.T @ np.linalg.solve(A, e_weighted)

            step = float(np.abs(dq).max())
            if step > self.cfg.max_iteration_step_rad:
                dq *= self.cfg.max_iteration_step_rad / step
            q = np.clip(q + dq, self._lo_rad, self._hi_rad)

        raw = np.rad2deg(q)

        commanded = raw.copy()
        if rate_limit:
            reference = seed if rate_reference_deg is None else np.asarray(rate_reference_deg, float)
            reference = reference[: self._n]
            delta = commanded - reference
            rate_clamped = np.abs(delta) > self.teleop.max_joint_step_deg
            if rate_clamped.any():
                commanded = reference + np.clip(
                    delta, -self.teleop.max_joint_step_deg, self.teleop.max_joint_step_deg
                )
        else:
            rate_clamped = np.zeros(self._n, dtype=bool)

        # Re-clamp regardless of what the iteration did: the rate limiter works
        # from a reference that may itself sit outside the limits when a caller
        # passes a measurement taken from a miscalibrated arm.
        limited = np.clip(commanded, self._lo, self._hi)
        limit_clamped = ~np.isclose(limited, commanded)
        commanded = limited

        readout = self.task_pose(commanded)
        achieved = readout.pose
        position_error_m = float(np.linalg.norm(target.position - achieved.position))
        pitch_error_rad = float(target.pitch - achieved.pitch)
        roll_error_rad = float(wrap_to_pi(target.roll - achieved.roll))
        command_converged = (
            position_error_m <= self.cfg.position_tolerance_m
            and abs(pitch_error_rad) <= self.cfg.angle_tolerance_rad
            and abs(roll_error_rad) <= self.cfg.angle_tolerance_rad
        )
        return IK5DResult(
            joints_deg=commanded,
            raw_joints_deg=raw,
            achieved=achieved,
            achieved_yaw_rad=readout.yaw,
            achieved_yaw_residual_rad=readout.yaw_residual,
            achieved_pose_world=self.forward(commanded),
            position_error_m=position_error_m,
            pitch_error_rad=pitch_error_rad,
            roll_error_rad=roll_error_rad,
            iterations=iterations,
            converged=command_converged,
            solver_converged=converged,
            min_singular_value=s_min,
            damping=damping,
            limit_clamped=limit_clamped,
            rate_clamped=rate_clamped,
        )
