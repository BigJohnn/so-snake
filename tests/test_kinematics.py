"""M3 verification: frame conventions, 5D task chart, projection, and IK."""

from __future__ import annotations

import numpy as np
import pytest

from so_snake.config import ArmConfig, TaskLimits, TeleopConfig
from so_snake.m3_safety import (
    OrientationProjector,
    SO100TaskPose,
    TaskFrame,
    TaskIK5D,
    TaskPoseTracker,
    tool_rotation,
)


@pytest.fixture(scope="module")
def ik() -> TaskIK5D:
    return TaskIK5D()


def test_world_frame_puts_the_arm_forward_along_x(ik: TaskIK5D) -> None:
    pose = ik.forward(np.zeros(5))
    x, y, z = pose[:3, 3]

    assert x > 0.2, f"TCP should be well forward on +X, got x={x:.4f}"
    assert abs(y) < 0.01, f"TCP should be near the sagittal plane, got y={y:.4f}"
    assert z > 0.0, f"TCP should be above the base plane, got z={z:.4f}"


def test_base_and_world_transforms_are_inverses() -> None:
    arm = ArmConfig()
    assert np.allclose(arm.world_from_base() @ arm.base_from_world(), np.eye(4), atol=1e-12)


def test_shoulder_pan_rotates_the_tcp_about_a_vertical_axis(ik: TaskIK5D) -> None:
    pans = [-30.0, 0.0, 30.0, 60.0]
    pts = np.array([ik.forward(np.array([p, 90.0, -90.0, 0.0, 0.0]))[:3, 3] for p in pans])

    assert np.ptp(pts[:, 2]) < 1e-5, "pan must not change TCP height"

    A = 2 * np.array([pts[1, :2] - pts[0, :2], pts[2, :2] - pts[0, :2]])
    b = np.array([
        pts[1, :2] @ pts[1, :2] - pts[0, :2] @ pts[0, :2],
        pts[2, :2] @ pts[2, :2] - pts[0, :2] @ pts[0, :2],
    ])
    centre = np.linalg.solve(A, b)

    assert np.allclose(centre, [0.0452, 0.0], atol=1e-4), f"pan axis at {centre}"
    radii = np.linalg.norm(pts[:, :2] - centre, axis=1)
    assert np.ptp(radii) < 1e-6, f"pan must preserve reach, radii {radii}"


def test_task_frame_round_trips_position_anchored_chart(ik: TaskIK5D) -> None:
    frame = ik.frame
    pose = SO100TaskPose(0.270, 0.060, 0.140, -0.6, 1.2)
    readout = frame.read(frame.to_pose_matrix(pose))

    assert np.allclose(readout.pose.as_array(), pose.as_array(), atol=1e-12)
    assert readout.yaw == pytest.approx(frame.yaw_at(pose.position), abs=1e-12)
    assert abs(readout.yaw_residual) < 1e-12


def test_position_yaw_uses_the_pan_axis_not_world_origin(ik: TaskIK5D) -> None:
    frame = ik.frame
    position = np.array([0.2452, 0.2, 0.13])

    assert frame.yaw_at(position) == pytest.approx(np.pi / 4, abs=1e-4)


def test_projection_rejects_uncontrollable_yaw_when_tool_is_horizontal(ik: TaskIK5D) -> None:
    pose = SO100TaskPose(0.270, 0.0, 0.140, 0.0, 0.0)
    projected = OrientationProjector(ik.chain, ik.frame).project(np.array([0.0, 0.0, 0.2]), pose)

    assert projected.delta_pitch == pytest.approx(0.0, abs=1e-12)
    assert projected.delta_roll == pytest.approx(0.0, abs=1e-12)
    assert projected.rejected_norm == pytest.approx(0.2, abs=1e-12)


def test_projection_turns_world_yaw_into_roll_when_tool_points_down(ik: TaskIK5D) -> None:
    pose = SO100TaskPose(0.270, 0.0, 0.140, -np.pi / 2, 0.0)
    projected = OrientationProjector(ik.chain, ik.frame).project(np.array([0.0, 0.0, 0.2]), pose)

    assert projected.delta_pitch == pytest.approx(0.0, abs=1e-12)
    assert abs(projected.delta_roll) == pytest.approx(0.2, abs=1e-12)
    assert projected.rejected_norm == pytest.approx(0.0, abs=1e-12)


def test_chart_projection_matches_jacobian_controllable_plane_in_workspace(ik: TaskIK5D) -> None:
    rng = np.random.default_rng(3)
    lo, hi = ArmConfig().limits_deg_array()
    limits = TaskLimits()
    q = []
    while len(q) < 40:
        candidate = rng.uniform(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo))
        position = ik.task_pose(candidate).pose.position
        if np.all(position >= np.asarray(limits.pos_min_m)) and np.all(position <= np.asarray(limits.pos_max_m)):
            q.append(candidate)
    projector = OrientationProjector(ik.chain, ik.frame)

    disagreement = np.array([projector.basis_disagreement_deg(joints) for joints in q])

    assert np.median(disagreement) < 2.0
    assert disagreement.max() < 3.5


def test_ik_round_trips_reachable_task_poses(ik: TaskIK5D) -> None:
    rng = np.random.default_rng(0)
    lo, hi = ArmConfig().limits_deg_array()
    limits = TaskLimits()

    pos_errors = []
    angle_errors = []
    while len(pos_errors) < 80:
        q_true = rng.uniform(lo + 10.0, hi - 10.0)
        target = ik.task_pose(q_true).pose
        if not (
            np.all(target.position >= np.asarray(limits.pos_min_m))
            and np.all(target.position <= np.asarray(limits.pos_max_m))
        ):
            continue
        seed = np.clip(q_true + rng.normal(0, 5.0, size=5), lo, hi)
        result = ik.solve(seed, target, rate_limit=False)
        pos_errors.append(result.position_error_m)
        angle_errors.append(result.orientation_error_rad)

    pos_errors = np.array(pos_errors) * 1000.0
    angle_errors = np.degrees(angle_errors)
    success = (pos_errors < 0.05) & (angle_errors < 0.02)

    assert success.mean() >= 0.99
    assert np.median(pos_errors) < 0.005
    assert np.median(angle_errors) < 0.002


def test_ik_cold_seed_solves_reachable_task_poses(ik: TaskIK5D) -> None:
    rng = np.random.default_rng(4)
    lo, hi = ArmConfig().limits_deg_array()
    limits = TaskLimits()
    failures = 0
    trials = 0
    while trials < 100:
        q_true = rng.uniform(lo + 10.0, hi - 10.0)
        target = ik.task_pose(q_true).pose
        if not (
            np.all(target.position >= np.asarray(limits.pos_min_m))
            and np.all(target.position <= np.asarray(limits.pos_max_m))
        ):
            continue
        trials += 1
        result = ik.solve(ik.seed_for(target), target, rate_limit=False)
        failures += int(result.position_error_m > 1e-3 or result.orientation_error_rad > np.deg2rad(1.0))

    assert failures == 0


def test_ik_output_always_respects_joint_limits(ik: TaskIK5D) -> None:
    lo, hi = ArmConfig().limits_deg_array()
    rng = np.random.default_rng(1)

    for _ in range(40):
        position = rng.uniform([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])
        target = SO100TaskPose(
            float(position[0]),
            float(position[1]),
            float(position[2]),
            float(rng.uniform(-np.pi, np.pi)),
            float(rng.uniform(-np.pi, np.pi)),
        )
        result = ik.solve(rng.uniform(lo, hi), target, rate_limit=False)

        assert np.all(result.joints_deg >= lo - 1e-9), f"below limit: {result.joints_deg}"
        assert np.all(result.joints_deg <= hi + 1e-9), f"above limit: {result.joints_deg}"


def test_ik_rate_limit_caps_joint_motion(ik: TaskIK5D) -> None:
    teleop = TeleopConfig()
    start = np.array([0.0, 90.0, -90.0, 0.0, 0.0])
    far = ik.task_pose(np.array([100.0, 30.0, -170.0, 60.0, 90.0])).pose

    result = ik.solve(start, far, rate_limit=True)
    step = np.abs(result.joints_deg - start)

    assert np.all(step <= teleop.max_joint_step_deg + 1e-9), f"step {step} exceeds cap"
    assert result.rate_clamped.any(), "a far target should trip the rate limiter"
    assert not result.converged


class TestTaskPoseTracker:
    def test_starts_at_home_inside_the_workspace(self) -> None:
        tracker = TaskPoseTracker()
        limits = TaskLimits()
        assert np.all(tracker.pose.position >= np.asarray(limits.pos_min_m) - 1e-12)
        assert np.all(tracker.pose.position <= np.asarray(limits.pos_max_m) + 1e-12)

    def test_approaches_absolute_targets_with_step_limits(self) -> None:
        tracker = TaskPoseTracker(home=SO100TaskPose(0.250, 0.0, 0.130, 0.0, 0.0))
        target = SO100TaskPose(0.350, 0.0, 0.130, 1.0, 0.0)

        update = tracker.approach(target)

        assert update.step_rate_limited
        assert np.linalg.norm(tracker.pose.position - np.array([0.250, 0.0, 0.130])) <= (
            TaskLimits().max_step_pos_m + 1e-12
        )
        assert abs(tracker.pose.pitch) <= TaskLimits().max_step_rot_rad + 1e-12

    def test_clamps_at_the_workspace_wall_and_reports_it(self) -> None:
        tracker = TaskPoseTracker(home=SO100TaskPose(0.250, 0.0, 0.130, 0.0, 0.0))
        limits = TaskLimits()

        update = None
        for _ in range(2000):
            update = tracker.update(np.array([1.0, 0.0, 0.0]))

        assert tracker.pose.x == pytest.approx(limits.pos_max_m[0], abs=1e-9)
        assert update is not None
        assert update.position_clamped[0]
        assert update.any_clamped

    def test_wraps_roll_but_clamps_pitch(self) -> None:
        tracker = TaskPoseTracker(home=SO100TaskPose(0.250, 0.0, 0.130, 0.0, 0.0))
        tracker.set_pose(SO100TaskPose(0.250, 0.0, 0.130, 10.0, 4.0 * np.pi + 0.2))

        assert tracker.pose.pitch <= TaskLimits().pitch_max_rad
        assert tracker.pose.roll == pytest.approx(0.2, abs=1e-12)


def test_tool_rotation_round_trips_through_task_frame() -> None:
    frame = TaskFrame(pan_axis_xy=np.array([0.0452, 0.0]))
    pose = SO100TaskPose(0.250, 0.030, 0.140, -0.4, 0.7)
    T = np.eye(4)
    T[:3, 3] = pose.position
    T[:3, :3] = tool_rotation(frame.yaw_at(pose.position), pose.pitch, pose.roll)

    assert np.allclose(frame.read(T).pose.as_array(), pose.as_array(), atol=1e-12)
