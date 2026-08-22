"""Configuration for the SO-100 arm, its workspace, and the teleoperation loop.

Everything here is geometry and tuning that must survive the trip from the
offline mock setup to the real arm unchanged, so it lives in one place rather
than being scattered through the control code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Joint order is the SO-100 URDF's chain order, which is also the order
# lerobot's SOFollower reports and accepts motor values in.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"

# Straight from the SO-100 URDF <limit> tags, converted to degrees. The gripper
# is listed separately because it is commanded directly rather than through IK.
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-114.59, 114.59),
    "shoulder_lift": (0.0, 200.54),
    "elbow_flex": (-180.0, 0.0),
    "wrist_flex": (-143.24, 68.75),
    "wrist_roll": (-180.0, 180.0),
    "gripper": (-11.46, 114.59),
}


@dataclass(frozen=True)
class ArmConfig:
    """Kinematic description of the arm."""

    urdf_path: Path = REPO_ROOT / "assets" / "urdf" / "so100" / "so100.urdf"
    joint_names: tuple[str, ...] = ARM_JOINTS
    tcp_frame: str = "gripper_frame_link"
    joint_limits_deg: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(JOINT_LIMITS_DEG)
    )

    # The SO-100 URDF's base frame has Z up but the arm extending along -Y: at
    # zero configuration the TCP sits at (-0.0002, -0.2374, +0.0956). Every
    # convention we work against -- joycon-robotics' workspace box, and the
    # operator's intuition that "stick forward" means forward -- assumes +X is
    # forward, so we rotate the base frame by +90 deg about Z to get there.
    # Verify with: scripts/check_frames.py
    world_from_base_yaw_deg: float = 90.0

    # Nothing here weights orientation against position any more. The 5-DoF arm
    # is now given a 5-dimensional target, so there is no over-constrained pose
    # for the solver to compromise on -- see `m3_safety/ik5d.py` and
    # `docs/plan_5dof_task_space.md`. The fields that used to live here
    # (`ik_orientation_weight`, `ik_retry_tolerance_m`, `ik_max_retries`,
    # `ik_max_retry_jump_deg`) were all workarounds for that compromise or for
    # placo's local minima, and went with it.

    def world_from_base(self) -> np.ndarray:
        """4x4 transform taking a pose in the URDF base frame into the world frame."""
        yaw = np.deg2rad(self.world_from_base_yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return T

    def base_from_world(self) -> np.ndarray:
        """Inverse of :meth:`world_from_base`."""
        T = self.world_from_base()
        out = np.eye(4)
        out[:3, :3] = T[:3, :3].T
        out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
        return out

    def limits_deg_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper arm-joint limits as arrays in `joint_names` order."""
        lo = np.array([self.joint_limits_deg[j][0] for j in self.joint_names])
        hi = np.array([self.joint_limits_deg[j][1] for j in self.joint_names])
        return lo, hi


@dataclass(frozen=True)
class TaskLimits:
    """Coarse bounds on the commanded 5D task pose, in the world frame.

    Applied to the *target* before IK runs, following joycon-robotics' `glimit`.
    Clamping the target rather than the solution means the operator hits a soft
    wall they can feel and back away from, instead of the arm silently drifting
    into a pose IK cannot reach.

    Coarse is the operative word. This is a box, and the reachable set is not;
    the box is the cheap first filter, and `m3_safety.atlas` supplies the
    position-dependent pitch interval behind it.

    Replaces `WorkspaceConfig`, whose roll/pitch/yaw box was deleted rather than
    retuned. A box on Euler angles cannot describe what a 5-DoF arm can hold --
    yaw is determined by position, so no interval on it is meaningful -- and the
    inherited one excluded the home pose's own orientation.
    """

    # From `scripts/map_workspace.py`, which grids the box and solves
    # position-only IK at every point. The largest fully-reachable box, forced
    # symmetric about y = 0 because the arm is, measured 11.84 litres at
    # x [0.164, 0.380], y [-0.215, 0.215], z [0.074, 0.202]. The original 80 mm
    # TCP floor was too conservative in MuJoCo teleop: the gripper mesh still
    # had about 72 mm table clearance when the task target was already clamped.
    # Keep the coarse task box usable and let min_robot_mesh_z_m provide the
    # geometric table clearance check.
    #
    # joycon-robotics' inherited box -- x [0.125, 0.380], y [-0.4, 0.4],
    # z [0.046, 0.23] -- is not to be used even as a coarse filter: it measured
    # 84% reachable, and its |y| <= 0.4 m is geometrically impossible, since the
    # TCP sweeps a circle of radius 0.310 m about the shoulder_pan axis.
    pos_min_m: tuple[float, float, float] = (0.170, -0.200, 0.004)
    pos_max_m: tuple[float, float, float] = (0.360, 0.200, 0.200)

    # Elevation of the gripper's approach axis, positive up. Unlike the deleted
    # rpy box this is a real coordinate of the task space, and these are the
    # measured envelope: the widest pitch reached at *any* voxel of the position
    # box, from `scripts/build_feasibility_atlas.py` over 40 M samples.
    #
    # The envelope is nearly the full half-circle, so this clamp barely
    # constrains anything -- and that is correct. What pitch is available is a
    # strong function of *where*: a third of the box can point 75 degrees down
    # and an eighth can reach 85, while at the edge of reach the arm is nearly
    # straight and pitch is pinned within a few degrees. No single interval can
    # express that, which is why the real constraint is `m3_safety.atlas` and
    # this is only a sanity bound applied before the atlas is consulted.
    #
    # The interval available at *every* voxel, by contrast, is empty. Using that
    # as the clamp would forbid essentially every useful gripper attitude.
    pitch_min_rad: float = -1.5707  # -90.00 deg, straight down
    pitch_max_rad: float = 1.5629  # +89.55 deg

    # Roll is deliberately unbounded. It is periodic, `wrist_roll` spans the
    # full +/-180 deg, and the real limits are cabling, the wrist camera's lead
    # and self-collision -- none of which are in the URDF. Bound it once it has
    # been measured on hardware, not before.

    # Safety ceiling on how far one control step may move the target, whatever
    # the operator or policy asks for. Catches an IMU spike, or a dropped and
    # resumed connection replaying a large accumulated delta.
    max_step_pos_m: float = 0.02
    max_step_rot_rad: float = 0.15

    def clamp_position(self, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clamp a position; also return a per-axis mask of which axes were clamped."""
        lo = np.asarray(self.pos_min_m)
        hi = np.asarray(self.pos_max_m)
        clamped = np.clip(position, lo, hi)
        return clamped, ~np.isclose(clamped, position)

    def clamp_pitch(self, pitch: float) -> tuple[float, bool]:
        clamped = float(np.clip(pitch, self.pitch_min_rad, self.pitch_max_rad))
        return clamped, not np.isclose(clamped, pitch)

    def centre(self) -> np.ndarray:
        return (np.asarray(self.pos_min_m) + np.asarray(self.pos_max_m)) / 2.0


@dataclass(frozen=True)
class IK5DConfig:
    """Tuning for the damped least squares solve in `m3_safety.ik5d`."""

    # Iteration budget. A teleoperation step moves the target by ~1.5 mm and is
    # seeded from the previous solution, so it converges in a handful; the
    # budget only binds when something calls the solver cold, as the atlas and
    # the gate tests do.
    max_iterations: int = 60

    position_tolerance_m: float = 1e-5
    angle_tolerance_rad: float = 1e-4

    # How many metres of position error one radian of angle error is worth,
    # where the two have to be traded off. Only bites near a singularity, where
    # the damping forces a compromise; 0.05 means a degree of approach angle is
    # worth about 0.9 mm of placement, which is the right way round for grasping.
    angle_weight_m: float = 0.05

    # Damping schedule. `damping_min` is a numerical floor, small enough that
    # away from singularities the solve is effectively exact. `damping_max`
    # takes over as the smallest singular value of `J_task` falls below
    # `singular_threshold`, keeping `dq` bounded through the singularity instead
    # of letting the pseudo-inverse blow up.
    #
    # The threshold is set from measurement: `J_w . null(J_v)` has singular
    # values around 0.006-0.018, so `J_task` degrades meaningfully below ~0.01.
    damping_min: float = 1e-6
    damping_max: float = 0.02
    singular_threshold: float = 0.01

    # Cap on one Newton step, so a large initial error walks in rather than
    # overshooting into a different IK branch.
    max_iteration_step_rad: float = 0.25

    # Closed-loop recovery threshold. If the previous-command seed falls into a
    # bad local branch and misses position by more than this, the teleop loop
    # tries the target-derived seed once, while still applying the velocity cap
    # against the previous command.
    reseed_position_error_m: float = 0.005

    # Give up when an iteration reduces the weighted residual by less than this
    # fraction. Near a singularity the damping makes the last micrometres cost
    # tens of iterations they are not worth; the residual at that point is
    # already two orders of magnitude inside grasp tolerance.
    min_relative_progress: float = 1e-3


@dataclass(frozen=True)
class TeleopConfig:
    """Teleoperation loop tuning."""

    # Record at 60 Hz by default.  This preserves the short contact/gripper
    # events that are costly to reconstruct after the fact; export can create
    # a time-consistent 30 Hz training set when a lighter policy loop is wanted.
    control_hz: float = 60.0

    # Rest configuration, specified in joint space rather than as a TCP pose.
    #
    # A pose can be written down that the arm cannot hold: the previous default
    # of position (0.230, 0, 0.130) with identity orientation was 144.6 deg away
    # from the nearest orientation achievable there, so the arm could never
    # settle at home and every run started with a large tracking error. Joint
    # space has no such failure mode -- forward kinematics of any in-limit
    # configuration is reachable by construction.
    #
    # These values put the TCP near the middle of the practical teleop box with
    # the gripper angled downward.
    home_joints_deg: tuple[float, float, float, float, float] = (0.0, 70.0, -70.0, 30.0, 0.0)

    # Per-axis translation gain applied to the controller's normalised input,
    # in metres per control step. joycon-robotics integrates 1 mm per tick at
    # unit dof_speed; tune this against the physical controller when changing
    # the control-loop frequency.
    translation_step_m: tuple[float, float, float] = (0.0015, 0.0015, 0.0015)

    # Extra gain for manual translation. The left stick drives table-plane X/Y;
    # the right stick's vertical axis drives Z.
    stick_translation_gain: float = 5.0

    # Gain on the operator's projected rotation, applied to the `(pitch, roll)`
    # deltas that come out of `OrientationProjector`. Dimensionless: 1.0 means
    # the tool turns as far as the operator's wrist did.
    #
    # The old `rotation_step_rad = 0.004` (~7 deg/s at 30 Hz) was not a comfort
    # setting, it was a brake on a runaway: a 6-DoF orientation target could
    # integrate into poses the arm cannot hold, and slowing the input was what
    # kept it from getting there within a session. With the target confined to
    # the arm's real task space that failure mode does not exist, so this is
    # free to be tuned for feel. 1.0 -- direct, one-to-one -- is the honest
    # starting point; retune it against the controller, not against tracking
    # error.
    rotation_gain: float = 1.0

    # Ceiling on per-step joint motion after IK, in degrees. At 30 Hz, 6 deg per
    # step is 180 deg/s, comfortably under the STS3215's free-running speed
    # while still being fast enough that teleop does not feel sluggish.
    max_joint_step_deg: float = 6.0

    # MuJoCo-only geometric table clearance: the lowest moving robot mesh point
    # must stay this far above the table plane. TCP/tool z alone is not enough:
    # with some wrist pitches the jaw or wrist body can sit below the tool frame.
    min_robot_mesh_z_m: float = 0.025

    # How close a point-to-point move (`move_to_joints`) has to get before the
    # arm counts as arrived, and how far off means it is stuck rather than
    # merely imprecise.
    #
    # These are properties of the servo, not preferences. The STS3215 in
    # position mode is a proportional controller, and lerobot's `configure()`
    # halves its gain (`P_Coefficient` 32 -> 16, "to avoid shakiness"), so it
    # settles with a standing offset wherever gravity or friction loads it.
    # Measured on this bench, from `observation.state.joints_deg` minus
    # `action.joint.commanded_deg` while the arm sat still at the start pose:
    #
    #     shoulder_pan  2.7    shoulder_lift  1.2    elbow_flex  0.8
    #     wrist_flex    1.1    wrist_roll     0.9        (degrees, mean)
    #
    # The old 1.0 deg tolerance was therefore unreachable on a good day, and
    # `move_to_joints` reported every homing and every replay approach as
    # "did not converge" -- which the replayer treated as fatal. 3.0 clears the
    # measured offsets; `joint_stuck_deg` is the separate question of whether
    # something is physically holding the arm, and only that should stop a move.
    joint_settle_tol_deg: float = 3.0
    joint_stuck_deg: float = 8.0

    gripper_open_deg: float = 90.0
    gripper_closed_deg: float = 2.0
    gripper_step_deg: float = 6.0


@dataclass(frozen=True)
class VideoConfig:
    """How camera frames are encoded into an episode.

    There is no globally right encoder, so the default decides per machine.
    Measured here on two 1080p streams at 30 Hz: `hevc_videotoolbox` needs 0.30
    of a core and 1.67 Mbps; `libsvtav1` needs 1.44 cores and 0.52 Mbps, at the
    same measured picture (43.0 vs 42.8 dB). Hardware is 4.8x cheaper in CPU,
    software 3.2x cheaper on disk, so the answer is whichever the recording
    machine is short of. `so_snake.data.video.select_encoder` applies the rule
    and writes both the choice and its reason into the episode.
    """

    # "auto" probes this machine. Anything else names an encoder, which is
    # probed too -- an operator who asks for one that cannot run here wants to
    # be told, not quietly given a different one.
    codec: str = "auto"

    # Below this many CPUs, "auto" prefers hardware. The software encoder needs
    # ~1.4 cores for two 1080p streams; against a control loop that needs 0.03,
    # that is affordable on eight cores and not on four.
    hw_core_threshold: int = 8

    # Software quality. CRF 30 measured 42.8 dB at 0.52 Mbps on camera footage.
    crf: int = 30
    # SVT-AV1 speed preset; 12 is the fastest and the only one that keeps up
    # with two cameras (141 fps against the 60 needed; libx264 manages 53).
    preset: int = 12

    # Hardware quality on VideoToolbox's 0-100 scale, used when `hw_bitrate` is
    # zero. Bitrate turned out to be the more predictable knob of the two.
    hw_quality: int = 50
    hw_bitrate: int = 1_500_000

    # Frames in flight per camera before frames start being dropped. Two
    # seconds at 30 Hz: long enough to ride out a keyframe, short enough that a
    # wedged encoder shows up while the take is still running.
    queue_size: int = 60


@dataclass(frozen=True)
class SoSnakeConfig:
    arm: ArmConfig = field(default_factory=ArmConfig)
    limits: TaskLimits = field(default_factory=TaskLimits)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    ik: IK5DConfig = field(default_factory=IK5DConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
