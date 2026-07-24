"""The Phase 0 teleoperation loop, in the arm's five-dimensional task space.

    controller frame
      -> clutch retargeter        raw sticks + IMU  ->  absolute 5D target
      -> task pose tracker        box clamp, per-step ceiling
      -> feasibility atlas        position-dependent pitch interval
      -> 5D task IK               five joints for five coordinates
      -> joint safety             limits, per-step velocity cap
      -> backend

which is the M3 chain of `docs/plan_5dof_task_space.md` with the M4 boundary at
the end: no 6-DoF end-effector pose crosses it, and nothing below the IK knows
what a task pose is.

Deliberately backend-agnostic. The same loop runs against `MockFollower` or
`MujocoBackend` with a `ScriptedSource` offline and against `SOFollowerBackend`
with a `NintendoProSource` on hardware, so what we debug at a desk is what runs
on the robot.

Every step records all three action streams of the dataset layout — raw device
frame, projected task action, executed joint command — so that the chain *raw
intent -> projected intent -> executed joints* stays auditable offline, and
changing the projector or the IK later does not mean re-recording.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..config import SoSnakeConfig
from ..m3_safety.atlas import DEFAULT_ATLAS_PATH, FeasibilityAtlas
from ..m3_safety.ik5d import TaskIK5D
from ..m3_safety.projection import OrientationProjector
from ..m3_safety.task_pose import SO100TaskPose, TaskFrame, TaskPoseTracker, wrap_to_pi
from ..m4_execution.backends import RobotBackend
from .clutch import ClutchRetargeter
from .sources import NintendoProSample, TeleopSource


@dataclass
class StepRecord:
    """Per-step telemetry: the dataset's three action streams, plus diagnostics."""

    index: int
    t: float

    # action.raw.* — exactly what the device reported
    raw: dict

    # action.task.* — the policy's training target
    task_target: np.ndarray  # (5,) absolute target after clamping
    task_delta: np.ndarray  # (5,) what this step actually moved it by
    gripper_cmd_deg: float

    # action.joint.* — what was sent to the servos
    commanded_joints_deg: np.ndarray
    measured_joints_deg: np.ndarray

    # observation.state.*
    achieved_task_pose: np.ndarray  # (5,)
    achieved_position: np.ndarray  # (3,)
    achieved_quaternion: np.ndarray  # (4,) full 6-DoF pose is kept, deliberately

    # M3 diagnostics — the plan's `orientation_projection_feedback`, demoted out
    # of the control loop but still logged.
    ik_position_error_m: float
    ik_pitch_error_rad: float
    ik_roll_error_rad: float
    projected_pitch_delta: float
    projected_roll_delta: float
    rejected_rotation_norm: float
    yaw_residual_rad: float
    orientation_saturated: bool
    workspace_clamped: bool
    atlas_pitch_clamped: bool
    atlas_roll_infeasible: bool
    joint_limit_clamped: bool
    joint_rate_clamped: bool
    command_safety_held: bool
    command_safety_reason: str
    robot_mesh_min_z_m: float | None
    robot_mesh_min_body: str
    ik_converged: bool
    ik_solver_converged: bool
    ik_reseeded: bool
    ik_iterations: int
    ik_min_singular_value: float
    clutch_engaged: bool
    loop_dt_s: float


@dataclass
class LoopStats:
    """Aggregates a run into the numbers the blueprint asks M3/M4 to report."""

    records: list[StepRecord] = field(default_factory=list)

    def add(self, record: StepRecord) -> None:
        self.records.append(record)

    def summary(self) -> dict[str, float]:
        if not self.records:
            return {}
        position = np.array([r.ik_position_error_m for r in self.records]) * 1000.0
        pitch = np.degrees(np.abs([r.ik_pitch_error_rad for r in self.records]))
        roll = np.degrees(np.abs([r.ik_roll_error_rad for r in self.records]))
        rejected = np.degrees([r.rejected_rotation_norm for r in self.records])
        residual = np.degrees(np.abs([r.yaw_residual_rad for r in self.records]))
        dts = np.array([r.loop_dt_s for r in self.records if r.loop_dt_s > 0])
        n = len(self.records)

        def fraction(predicate) -> float:
            return float(sum(bool(predicate(r)) for r in self.records) / n)

        return {
            "steps": float(n),
            "ik_pos_err_median_mm": float(np.median(position)),
            "ik_pos_err_p95_mm": float(np.percentile(position, 95)),
            "ik_pos_err_max_mm": float(position.max()),
            "ik_pitch_err_p95_deg": float(np.percentile(pitch, 95)),
            "ik_roll_err_p95_deg": float(np.percentile(roll, 95)),
            "ik_converged_frac": fraction(lambda r: r.ik_converged),
            "ik_solver_converged_frac": fraction(lambda r: r.ik_solver_converged),
            "ik_reseeded_frac": fraction(lambda r: r.ik_reseeded),
            "yaw_residual_p95_deg": float(np.percentile(residual, 95)),
            "rejected_rotation_p95_deg": float(np.percentile(rejected, 95)),
            "workspace_clamped_frac": fraction(lambda r: r.workspace_clamped),
            "atlas_pitch_clamped_frac": fraction(lambda r: r.atlas_pitch_clamped),
            "joint_limit_clamped_frac": fraction(lambda r: r.joint_limit_clamped),
            "joint_rate_clamped_frac": fraction(lambda r: r.joint_rate_clamped),
            "loop_hz_median": float(1.0 / np.median(dts)) if len(dts) else float("nan"),
            "loop_hz_p05": float(1.0 / np.percentile(dts, 95)) if len(dts) else float("nan"),
        }


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """`(w, x, y, z)`, for logging the full 6-DoF pose alongside the 5D one."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1], q[j + 1], q[k + 1] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return q


class TeleopLoop:
    """Drives one arm from one teleoperation source."""

    def __init__(
        self,
        source: TeleopSource,
        backend: RobotBackend,
        config: SoSnakeConfig | None = None,
        ik: TaskIK5D | None = None,
        atlas: FeasibilityAtlas | None = None,
    ) -> None:
        self.config = config or SoSnakeConfig()
        self.source = source
        self.backend = backend
        self.ik = ik or TaskIK5D(arm=self.config.arm, teleop=self.config.teleop, ik=self.config.ik)
        self.frame: TaskFrame = self.ik.frame
        self.projector = OrientationProjector(self.ik.chain, self.frame)
        self.retargeter = ClutchRetargeter(self.projector, self.config.teleop)

        if atlas is None and DEFAULT_ATLAS_PATH.exists():
            atlas = FeasibilityAtlas.load(DEFAULT_ATLAS_PATH)
        self.atlas = atlas

        self._n_arm = len(self.config.arm.joint_names)
        home_readout = self.ik.task_pose(np.asarray(self.config.teleop.home_joints_deg, float))
        self.tracker = TaskPoseTracker(self.config.limits, home=home_readout.pose)
        self.stats = LoopStats()

    def _gripper_deg(self, opening: float) -> float:
        teleop = self.config.teleop
        return teleop.gripper_closed_deg + float(np.clip(opening, 0.0, 1.0)) * (
            teleop.gripper_open_deg - teleop.gripper_closed_deg
        )

    def measured_task_pose(self) -> SO100TaskPose:
        """The arm's task pose from forward kinematics of the joints read back."""
        measured = self.backend.read_joints_deg()
        return self.ik.task_pose(measured[: self._n_arm]).pose

    def sync_target_to_arm(self) -> None:
        """Adopt the arm's current task pose as the target.

        Called before the first step so the arm does not lurch from wherever it
        happens to be to the configured home pose the instant teleop starts.
        """
        self.tracker.set_pose(self.measured_task_pose())
        self.retargeter.reset()

    def run(self, max_steps: int | None = None, realtime: bool = True) -> LoopStats:
        """Run until the source signals stop, or `max_steps` steps have elapsed.

        Args:
            max_steps: Stop after this many steps. None runs until the source stops.
            realtime: Sleep to hold `control_hz`. Disable for offline runs, where
                pacing to wall-clock only makes the test slow.
        """
        period = 1.0 / self.config.teleop.control_hz

        if not self.backend.is_connected:
            self.backend.connect()
        if not self.source.is_connected:
            self.source.connect()

        self.sync_target_to_arm()

        last_command: np.ndarray | None = None
        last_gripper_command: float | None = None
        t_start = time.perf_counter()
        t_prev = t_start
        step = 0

        while max_steps is None or step < max_steps:
            t_loop = time.perf_counter()
            sample: NintendoProSample = self.source.read()
            if "stop" in sample.events:
                break

            measured = self.backend.read_joints_deg()
            arm_measured = measured[: self._n_arm]
            measured_pose = self.ik.task_pose(arm_measured).pose

            if "reset" in sample.events:
                self.tracker.reset()
                self.retargeter.reset()

            retarget = self.retargeter.update(sample, measured_pose)

            previous_target = self.tracker.pose
            update = self.tracker.approach(retarget.target)

            target = update.pose
            atlas_pitch_clamped = False
            atlas_roll_infeasible = False
            if self.atlas is not None:
                projection = self.atlas.project(target)
                target = projection.pose
                atlas_pitch_clamped = projection.pitch_clamped
                atlas_roll_infeasible = projection.roll_infeasible
                # Write the atlas's verdict back into the target, so the next
                # step integrates from something reachable rather than from a
                # request that was silently overruled.
                self.tracker.pose = target

            # Seed from the last command, not the measurement: the servo lags,
            # so seeding from where it currently is drags the solution backwards
            # and lets successive solves drift between IK branches.
            seed = arm_measured if last_command is None else last_command
            result = self.ik.solve(seed, target, rate_reference_deg=seed)
            ik_reseeded = False
            if (
                not result.solver_converged
                and result.position_error_m > self.config.ik.reseed_position_error_m
            ):
                # Staying on the previous-command branch is normally what keeps
                # teleop continuous, but near a joint limit the local solve can
                # get trapped on a branch that is simply the wrong side of the
                # elbow. Re-seed from the target chart only as a recovery path;
                # the rate cap is still measured against the previous command.
                recovered = self.ik.solve(
                    self.ik.seed_for(target),
                    target,
                    rate_reference_deg=seed,
                )
                if (
                    recovered.position_error_m < result.position_error_m
                    or recovered.orientation_error_rad < result.orientation_error_rad
                ):
                    result = recovered
                    ik_reseeded = True
            commanded_joints_deg = result.joints_deg
            achieved = result.achieved
            achieved_pose_world = result.achieved_pose_world
            position_error_m = result.position_error_m
            pitch_error_rad = result.pitch_error_rad
            roll_error_rad = result.roll_error_rad
            yaw_residual_rad = result.achieved_yaw_residual_rad
            ik_converged = result.converged
            command_safety_held = False
            command_safety_reason = ""
            robot_mesh_min_z_m: float | None = None
            robot_mesh_min_body = ""
            gripper_deg = self._gripper_deg(sample.gripper)
            seed_gripper = float(measured[self._n_arm]) if last_gripper_command is None else last_gripper_command

            def hold_previous_command(reason: str) -> None:
                nonlocal commanded_joints_deg, achieved, achieved_pose_world, position_error_m
                nonlocal pitch_error_rad, roll_error_rad, yaw_residual_rad, ik_converged
                nonlocal command_safety_held, command_safety_reason, gripper_deg
                command_safety_held = True
                command_safety_reason = reason
                commanded_joints_deg = np.asarray(seed, float).copy()
                gripper_deg = seed_gripper
                readout = self.ik.task_pose(commanded_joints_deg)
                achieved = readout.pose
                achieved_pose_world = self.ik.forward(commanded_joints_deg)
                position_error_m = float(np.linalg.norm(target.position - achieved.position))
                pitch_error_rad = float(target.pitch - achieved.pitch)
                roll_error_rad = float(wrap_to_pi(target.roll - achieved.roll))
                yaw_residual_rad = readout.yaw_residual
                ik_converged = False
                self.tracker.set_pose(achieved)
                self.retargeter.force_target(achieved, sample)

            z_floor = float(self.config.limits.pos_min_m[2])
            if float(achieved.position[2]) < z_floor:
                hold_previous_command("post_rate_achieved_z_below_workspace_floor")

            clearance_probe = getattr(self.backend, "command_robot_mesh_min_z_deg", None)
            if callable(clearance_probe):
                command = np.concatenate([commanded_joints_deg, [gripper_deg]])
                robot_mesh_min_z_m, robot_mesh_min_body = clearance_probe(command)
                if robot_mesh_min_z_m < self.config.teleop.min_robot_mesh_z_m:
                    hold_previous_command(
                        "post_rate_robot_mesh_below_clearance:"
                        f"{robot_mesh_min_body}:{robot_mesh_min_z_m:.4f}"
                    )
                    command = np.concatenate([commanded_joints_deg, [gripper_deg]])
                    robot_mesh_min_z_m, robot_mesh_min_body = clearance_probe(command)

            last_command = commanded_joints_deg
            last_gripper_command = gripper_deg

            self.backend.write_joints_deg(np.concatenate([commanded_joints_deg, [gripper_deg]]))

            now = time.perf_counter()
            self.stats.add(
                StepRecord(
                    index=step,
                    t=now - t_start,
                    raw=sample.as_log_record(),
                    task_target=target.as_array(),
                    task_delta=target.as_array() - previous_target.as_array(),
                    gripper_cmd_deg=gripper_deg,
                    commanded_joints_deg=commanded_joints_deg,
                    measured_joints_deg=arm_measured,
                    achieved_task_pose=achieved.as_array(),
                    achieved_position=achieved_pose_world[:3, 3],
                    achieved_quaternion=_rotation_to_quaternion(achieved_pose_world[:3, :3]),
                    ik_position_error_m=position_error_m,
                    ik_pitch_error_rad=pitch_error_rad,
                    ik_roll_error_rad=roll_error_rad,
                    projected_pitch_delta=retarget.projected_pitch_delta,
                    projected_roll_delta=retarget.projected_roll_delta,
                    rejected_rotation_norm=retarget.rejected_rotation_norm,
                    yaw_residual_rad=yaw_residual_rad,
                    orientation_saturated=bool(update.pitch_clamped or atlas_pitch_clamped),
                    workspace_clamped=bool(update.position_clamped.any()),
                    atlas_pitch_clamped=atlas_pitch_clamped,
                    atlas_roll_infeasible=atlas_roll_infeasible,
                    joint_limit_clamped=bool(result.limit_clamped.any()),
                    joint_rate_clamped=bool(result.rate_clamped.any()),
                    command_safety_held=command_safety_held,
                    command_safety_reason=command_safety_reason,
                    robot_mesh_min_z_m=robot_mesh_min_z_m,
                    robot_mesh_min_body=robot_mesh_min_body,
                    ik_converged=ik_converged,
                    ik_solver_converged=result.solver_converged,
                    ik_reseeded=ik_reseeded,
                    ik_iterations=result.iterations,
                    ik_min_singular_value=result.min_singular_value,
                    clutch_engaged=retarget.engaged,
                    loop_dt_s=now - t_prev,
                )
            )
            t_prev = now
            step += 1

            if realtime:
                slack = period - (time.perf_counter() - t_loop)
                if slack > 0:
                    time.sleep(slack)

        return self.stats

    def to_records(self) -> list[dict]:
        """The run as plain dictionaries, ready for a dataset writer."""
        return [asdict(record) for record in self.stats.records]
