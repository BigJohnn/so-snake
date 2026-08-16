"""Replay: driving an arm from a recorded episode.

Two modes, and the difference between them is the reason this module exists at
all rather than being a `for` loop over `write_joints_deg`:

  **joint** — send the recorded `action.joint.commanded_deg` back out. This asks
    *does the arm reproduce what it did?* Nothing is recomputed, so a divergence
    is the hardware's, the servo tuning's, or the scene's -- not the solver's.
    This is the mode for validating a demonstration, and the only one that is
    meaningful on the real arm without further thought.

  **task** — take the recorded `action.task.target`, the 5D pose, and solve it
    again through the *current* IK and feasibility atlas. This asks *would
    today's controller have done the same thing?* It is how a change to the
    projector, the atlas or the solver is regression-tested against real
    operator input, which is what `TeleopLoop`'s docstring promises when it says
    changing the IK later does not mean re-recording.

## Safety, which is not optional here

A recording is not a safe command sequence on its own. The arm starts wherever
it happens to be, the playback rate is under the operator's control, and the
episode may have been recorded under a different workspace box. So:

  * playback begins with a rate-limited, IK-free **approach** to the episode's
    first pose, via `move_to_joints`, rather than jumping there;
  * every commanded step is clamped to a **joint velocity** in deg/s, not deg
    per step -- otherwise asking for 2x speed would silently double the real
    speed of the arm while every per-step check still passed;
  * commands are clamped to the arm's joint limits, from the *current* config,
    since those describe the physical arm and not the recording;
  * where the backend can tell us (MuJoCo), the **mesh clearance** is checked
    before the write, and the previous command held if the frame would put a
    link through the table -- the same rule the teleop loop applies.

`inspect_episode` runs those checks statically, before anything moves. On the
real arm, run it first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import SoSnakeConfig
from ..m3_safety.atlas import DEFAULT_ATLAS_PATH, FeasibilityAtlas
from ..m3_safety.ik5d import TaskIK5D
from ..m3_safety.task_pose import SO100TaskPose, wrap_to_pi
from ..m4_execution.backends import RobotBackend
from ..m4_execution.motion import move_to_joints
from ..pacing import RateKeeper
from .episode import Episode

REPLAY_MODES = ("joint", "task")


@dataclass(frozen=True)
class ReplayConfig:
    """How to play an episode back."""

    mode: str = "joint"

    # Playback rate multiplier. Above 1.0 the arm genuinely moves faster, which
    # is why the velocity cap below is expressed in deg/s and not deg/step.
    speed: float = 1.0

    # Joint velocity ceiling. Defaults to the teleoperation loop's own budget
    # (`max_joint_step_deg` at `control_hz`), so a 1x replay of an episode
    # recorded by that loop is never clamped and anything faster is.
    max_joint_rate_deg_s: float | None = None

    # Approach to the first frame: how far the command leads the measurement per
    # step, how close counts as arrived, and how far off means the arm is stuck.
    #
    # None takes both from `TeleopConfig` (`joint_settle_tol_deg` /
    # `joint_stuck_deg`), which is where the servo's measured standing offset
    # lives. They were 1.0 deg and "any miss is fatal" here, and on the real arm
    # that combination aborted every replay before it played a frame: the
    # shoulder settles ~2.7 deg out and cannot do better, so the approach never
    # reported success and the replay gave up -- releasing torque as it went.
    approach_step_deg: float = 6.0
    approach_tol_deg: float | None = None
    approach_stuck_deg: float | None = None

    realtime: bool = True
    check_clearance: bool = True

    def __post_init__(self) -> None:
        if self.mode not in REPLAY_MODES:
            raise ValueError(f"mode must be one of {REPLAY_MODES}, got {self.mode!r}")
        if not 0.05 <= self.speed <= 4.0:
            raise ValueError(f"speed must be within [0.05, 4.0], got {self.speed}")

    def rate_limit_deg_s(self, config: SoSnakeConfig) -> float:
        if self.max_joint_rate_deg_s is not None:
            return float(self.max_joint_rate_deg_s)
        return float(config.teleop.max_joint_step_deg * config.teleop.control_hz)

    def approach_tolerance_deg(self, config: SoSnakeConfig) -> float:
        if self.approach_tol_deg is not None:
            return float(self.approach_tol_deg)
        return float(config.teleop.joint_settle_tol_deg)

    def approach_stuck_tolerance_deg(self, config: SoSnakeConfig) -> float:
        if self.approach_stuck_deg is not None:
            return float(self.approach_stuck_deg)
        return float(config.teleop.joint_stuck_deg)


@dataclass
class ReplayStep:
    """One replayed frame: what was asked, what was sent, what came back."""

    index: int
    t: float

    recorded_joints_deg: np.ndarray  # (5,) from the episode
    commanded_joints_deg: np.ndarray  # (5,) after rate/limit/clearance clamps
    measured_joints_deg: np.ndarray  # (5,) read back before the write
    gripper_cmd_deg: float

    # Deviation of the *commanded* stream from the recording. Non-zero only when
    # a clamp bit, or when task mode re-solved to a different configuration.
    command_deviation_deg: float

    # Deviation of the *measured* stream from the recording. This is the tracking
    # question -- did the arm actually get there.
    tracking_error_deg: float

    task_target: np.ndarray  # (5,) from the episode
    achieved_task_pose: np.ndarray  # (5,) FK of what was commanded
    position_error_m: float
    pitch_error_rad: float
    roll_error_rad: float

    rate_clamped: bool
    limit_clamped: bool
    safety_held: bool
    safety_reason: str
    robot_mesh_min_z_m: float | None
    ik_converged: bool
    loop_dt_s: float


@dataclass
class ReplayReport:
    """The outcome of one replay."""

    episode_id: str
    mode: str
    n_steps: int = 0
    completed: bool = False
    aborted_reason: str = ""
    approach_reached: bool = False
    # How far off the first frame the approach ended, and which joints. Non-empty
    # `approach_note` means the replay went ahead anyway -- worth showing, not
    # worth stopping for.
    approach_residual_deg: float = 0.0
    approach_note: str = ""
    steps: list[ReplayStep] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if not self.steps:
            return {}
        command_dev = np.array([s.command_deviation_deg for s in self.steps])
        tracking = np.array([s.tracking_error_deg for s in self.steps])
        position = np.array([s.position_error_m for s in self.steps]) * 1000.0
        n = len(self.steps)
        return {
            "steps": float(n),
            "command_deviation_p95_deg": float(np.percentile(command_dev, 95)),
            "command_deviation_max_deg": float(command_dev.max()),
            "tracking_error_p95_deg": float(np.percentile(tracking, 95)),
            "tracking_error_max_deg": float(tracking.max()),
            "task_position_error_p95_mm": float(np.percentile(position, 95)),
            "task_position_error_max_mm": float(position.max()),
            "rate_clamped_frac": float(sum(s.rate_clamped for s in self.steps) / n),
            "limit_clamped_frac": float(sum(s.limit_clamped for s in self.steps) / n),
            "safety_held_frac": float(sum(s.safety_held for s in self.steps) / n),
            "ik_converged_frac": float(sum(s.ik_converged for s in self.steps) / n),
        }


@dataclass(frozen=True)
class Issue:
    """One thing found by `inspect_episode`."""

    level: str  # "error" blocks replay; "warning" is for the operator to weigh
    message: str


def inspect_episode(
    episode: Episode,
    config: SoSnakeConfig | None = None,
    *,
    backend_joint_names: tuple[str, ...] | None = None,
    target_physical: bool = False,
    replay: ReplayConfig | None = None,
) -> list[Issue]:
    """Check an episode against the current config before anything moves.

    Errors are contract violations -- a different arm, a different joint order,
    an unreadable file -- and mean the episode must not be replayed. Warnings
    are the operator's judgement call: an episode recorded in simulation being
    sent to the real arm, or one recorded under a workspace box that has since
    been tightened.

    `target_physical` says the replay would command a real arm, which is what
    makes several of these warnings worth raising at all. Set it from
    `RigSpec.is_physical`, not from the backend's class.
    """
    config = config or SoSnakeConfig()
    replay = replay or ReplayConfig()
    issues: list[Issue] = []

    if episode.meta.n_steps == 0 or len(episode.commanded_joints_deg) == 0:
        issues.append(Issue("error", "episode has no frames"))
        return issues

    arm_joints = list(config.arm.joint_names)
    recorded = [name for name in episode.meta.joint_names if name != "gripper"]
    if recorded and recorded != arm_joints:
        issues.append(
            Issue("error", f"joint order differs: episode {recorded} vs config {arm_joints}")
        )

    commanded = np.asarray(episode.commanded_joints_deg, float)
    if commanded.shape[1] != len(arm_joints):
        issues.append(
            Issue(
                "error",
                f"episode has {commanded.shape[1]} arm joints, config expects {len(arm_joints)}",
            )
        )
        return issues

    if backend_joint_names is not None:
        expected = (*arm_joints, "gripper")
        if tuple(backend_joint_names) != expected:
            issues.append(
                Issue("error", f"backend joints {tuple(backend_joint_names)} != expected {expected}")
            )

    lo, hi = config.arm.limits_deg_array()
    below = (commanded < lo - 1e-6).any(axis=0)
    above = (commanded > hi + 1e-6).any(axis=0)
    for i, name in enumerate(arm_joints):
        if below[i] or above[i]:
            issues.append(
                Issue(
                    "warning",
                    f"{name} was recorded outside the current limits "
                    f"[{lo[i]:.1f}, {hi[i]:.1f}]; it will be clamped on replay",
                )
            )

    recorded_hz = episode.meta.control_hz or config.teleop.control_hz
    playback_hz = recorded_hz * replay.speed
    step_deg = np.abs(np.diff(commanded, axis=0)).max() if len(commanded) > 1 else 0.0
    peak_rate = float(step_deg * playback_hz)
    limit = replay.rate_limit_deg_s(config)
    if peak_rate > limit:
        issues.append(
            Issue(
                "warning",
                f"peak joint rate at {replay.speed:g}x is {peak_rate:.0f} deg/s, over the "
                f"{limit:.0f} deg/s cap; those frames will be rate-clamped and the replay "
                f"will lag the recording",
            )
        )

    if episode.meta.simulated and target_physical:
        issues.append(
            Issue(
                "warning",
                "episode was recorded in simulation; nothing about it has been checked against "
                "the real scene, so clear the workspace before playing it on the arm",
            )
        )

    recorded_urdf = str(episode.meta.config.get("arm", {}).get("urdf_path", ""))
    if recorded_urdf and recorded_urdf != str(config.arm.urdf_path):
        issues.append(
            Issue("warning", f"recorded against a different URDF: {recorded_urdf}")
        )

    return issues


class EpisodeReplayer:
    """Plays one episode back to one backend."""

    def __init__(
        self,
        episode: Episode,
        backend: RobotBackend,
        config: SoSnakeConfig | None = None,
        replay: ReplayConfig | None = None,
        *,
        ik: TaskIK5D | None = None,
        atlas: FeasibilityAtlas | None = None,
    ) -> None:
        self.episode = episode
        self.backend = backend
        self.config = config or SoSnakeConfig()
        self.replay = replay or ReplayConfig()
        self._n_arm = len(self.config.arm.joint_names)
        self._lo, self._hi = self.config.arm.limits_deg_array()

        # Both modes need the solver's forward kinematics, to report where the
        # commanded joints actually put the tool -- that is the number the
        # operator judges a replay by, and it is not in the recording. Only task
        # mode also needs the atlas, since only it re-projects the target.
        self._ik = ik or TaskIK5D(arm=self.config.arm, teleop=self.config.teleop, ik=self.config.ik)
        self._atlas = atlas
        if self.replay.mode == "task" and self._atlas is None and DEFAULT_ATLAS_PATH.exists():
            self._atlas = FeasibilityAtlas.load(DEFAULT_ATLAS_PATH)

    def run(
        self,
        *,
        on_step: Callable[[ReplayStep], None] | None = None,
        should_continue: Callable[[], bool] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> ReplayReport:
        """Approach the first frame, then play the episode.

        `on_progress(phase, value)` reports the approach ("approach", remaining
        degrees) so a GUI can show that something is happening before the
        playback proper starts -- the approach can take several seconds and
        looks like a hang otherwise.
        """
        episode = self.episode
        report = ReplayReport(episode_id=episode.meta.id, mode=self.replay.mode)

        commanded = np.asarray(episode.commanded_joints_deg, float)
        gripper = np.asarray(episode.gripper_cmd_deg, float)
        targets = np.asarray(episode.task_target, float)
        n = len(commanded)
        if n == 0:
            report.aborted_reason = "episode has no frames"
            return report

        if not self.backend.is_connected:
            self.backend.connect()

        first = np.clip(np.concatenate([commanded[0], [gripper[0]]]), self._lo_full(), self._hi_full())
        approach = move_to_joints(
            self.backend,
            first,
            step_deg=self.replay.approach_step_deg,
            tol_deg=self.replay.approach_tolerance_deg(self.config),
            hz=self.config.teleop.control_hz,
            on_progress=(lambda remaining: on_progress("approach", remaining)) if on_progress else None,
            should_continue=should_continue,
            sleep=time.sleep if self.replay.realtime else (lambda _s: None),
        )
        report.approach_reached = bool(approach)
        report.approach_residual_deg = approach.max_residual_deg

        if approach.interrupted:
            report.aborted_reason = "stopped by the operator during the approach"
            return report
        if not approach.reached:
            # Two different failures used to share one verdict, and the strict
            # one was wrong: a servo that settles a couple of degrees out has
            # *arrived* as far as anything can tell it to, and refusing to play
            # the episode over that is how a real replay ended before it began.
            # Only a residual big enough to be an obstruction, a joint limit or
            # a stalled drive is worth stopping for -- and playback itself is
            # rate-limited, so it absorbs a small residual in its first frames.
            stuck_deg = self.replay.approach_stuck_tolerance_deg(self.config)
            if approach.max_residual_deg > stuck_deg:
                report.aborted_reason = (
                    f"the arm did not reach the first frame: {approach.describe()} "
                    f"(over the {stuck_deg:.0f} deg limit"
                    + (", and it had stopped moving" if approach.stalled else "")
                    + ") -- check for an obstruction, and that the joint map and the "
                    "episode belong to this arm"
                )
                return report
            report.approach_note = (
                f"started {approach.max_residual_deg:.1f} deg off the first frame "
                f"({approach.describe()}); within the {stuck_deg:.0f} deg limit, so the "
                "replay went ahead and the first frames close the rest"
            )

        # The rate the take *ran* at, not the one it was configured for. Those
        # differed by 15% for everything recorded before the pacing fix (see
        # `so_snake.pacing`): configured 30 Hz, achieved 26. Replaying such an
        # episode at its configured rate plays it back 15% fast, which on this
        # arm is a real difference -- the tracking lag is the largest term in
        # the motion, and it does not scale with playback speed.
        recorded_hz = episode.playback_hz or self.config.teleop.control_hz
        period = 1.0 / (recorded_hz * self.replay.speed)
        rate_limit = self.replay.rate_limit_deg_s(self.config)
        max_step_deg = rate_limit * period
        keeper = RateKeeper(
            recorded_hz * self.replay.speed, enabled=self.replay.realtime
        )

        clearance_probe = getattr(self.backend, "command_robot_mesh_min_z_deg", None)
        if not (self.replay.check_clearance and callable(clearance_probe)):
            clearance_probe = None

        last_command = np.asarray(commanded[0], float).copy()
        last_gripper = float(gripper[0])
        t_start = time.perf_counter()
        t_prev = t_start

        for i in range(n):
            if should_continue is not None and not should_continue():
                report.aborted_reason = "stopped by the operator"
                break

            measured = np.asarray(self.backend.read_joints_deg(), float)
            arm_measured = measured[: self._n_arm]

            wanted, ik_converged = self._frame_command(i, commanded, targets, last_command)

            limited = np.clip(wanted, last_command - max_step_deg, last_command + max_step_deg)
            rate_clamped = bool(np.any(np.abs(limited - wanted) > 1e-9))
            clamped = np.clip(limited, self._lo, self._hi)
            limit_clamped = bool(np.any(np.abs(clamped - limited) > 1e-9))

            gripper_deg = float(np.clip(gripper[i], *self.config.arm.joint_limits_deg["gripper"]))

            safety_held = False
            safety_reason = ""
            mesh_min_z: float | None = None
            if clearance_probe is not None:
                mesh_min_z, body = clearance_probe(np.concatenate([clamped, [gripper_deg]]))
                if mesh_min_z < self.config.teleop.min_robot_mesh_z_m:
                    safety_held = True
                    safety_reason = f"robot_mesh_below_clearance:{body}:{mesh_min_z:.4f}"
                    clamped = last_command.copy()
                    gripper_deg = last_gripper
                    mesh_min_z, _ = clearance_probe(np.concatenate([clamped, [gripper_deg]]))

            self.backend.write_joints_deg(np.concatenate([clamped, [gripper_deg]]))
            last_command, last_gripper = clamped.copy(), gripper_deg

            achieved = self._ik.task_pose(clamped).pose
            target_pose = SO100TaskPose.from_array(targets[i])
            now = time.perf_counter()

            step = ReplayStep(
                index=i,
                t=now - t_start,
                recorded_joints_deg=commanded[i].copy(),
                commanded_joints_deg=clamped.copy(),
                measured_joints_deg=arm_measured,
                gripper_cmd_deg=gripper_deg,
                command_deviation_deg=float(np.abs(clamped - commanded[i]).max()),
                tracking_error_deg=float(np.abs(arm_measured - commanded[i]).max()),
                task_target=targets[i].copy(),
                achieved_task_pose=achieved.as_array(),
                position_error_m=float(np.linalg.norm(target_pose.position - achieved.position)),
                pitch_error_rad=float(target_pose.pitch - achieved.pitch),
                roll_error_rad=float(wrap_to_pi(target_pose.roll - achieved.roll)),
                rate_clamped=rate_clamped,
                limit_clamped=limit_clamped,
                safety_held=safety_held,
                safety_reason=safety_reason,
                robot_mesh_min_z_m=mesh_min_z,
                ik_converged=ik_converged,
                loop_dt_s=now - t_prev,
            )
            report.steps.append(step)
            report.n_steps = len(report.steps)
            if on_step is not None:
                on_step(step)
            t_prev = now

            keeper.wait()
        else:
            report.completed = True

        return report

    # ---------------------------------------------------------------- helpers

    def _frame_command(
        self,
        i: int,
        commanded: np.ndarray,
        targets: np.ndarray,
        seed: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """The joints this frame asks for, before any clamping."""
        if self.replay.mode == "joint":
            return np.asarray(commanded[i], float), True

        target = SO100TaskPose.from_array(targets[i])
        if self._atlas is not None:
            target = self._atlas.project(target).pose
        result = self._ik.solve(seed, target, rate_reference_deg=seed)
        return result.joints_deg, bool(result.converged)

    def _lo_full(self) -> np.ndarray:
        return np.array(
            [self.config.arm.joint_limits_deg[j][0] for j in self.backend.joint_names]
        )

    def _hi_full(self) -> np.ndarray:
        return np.array(
            [self.config.arm.joint_limits_deg[j][1] for j in self.backend.joint_names]
        )
