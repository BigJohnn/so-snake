"""M4 verification: the 5D control loop end to end, with no hardware present."""

from __future__ import annotations

import numpy as np
import pytest

from so_snake.config import ArmConfig, SoSnakeConfig
from so_snake.m3_safety import TaskIK5D
from so_snake.m3_safety.ik5d import IK5DResult
from so_snake.m4_execution import MockFollower
from so_snake.teleop import NintendoProSample, ScriptedSource, TeleopLoop

IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0])


def sample(
    *,
    t: float = 0.0,
    left: tuple[float, float] = (0.0, 0.0),
    right: tuple[float, float] = (0.0, 0.0),
    clutch: bool = True,
    gripper: float = 1.0,
    events: frozenset[str] = frozenset(),
) -> NintendoProSample:
    return NintendoProSample(
        t=t,
        left_stick=np.array(left, dtype=float),
        right_stick=np.array(right, dtype=float),
        imu_quaternion=IDENTITY_QUATERNION.copy(),
        clutch=clutch,
        gripper=gripper,
        events=events,
    )


@pytest.fixture(scope="module")
def ik() -> TaskIK5D:
    return TaskIK5D()


def make_loop(source: ScriptedSource, ik: TaskIK5D, **backend_kwargs) -> TeleopLoop:
    backend = MockFollower(**backend_kwargs)
    return TeleopLoop(source, backend, SoSnakeConfig(), ik=ik)


class TestMockFollower:
    def test_reports_arm_joints_then_gripper(self) -> None:
        backend = MockFollower()
        assert backend.joint_names == (*ArmConfig().joint_names, "gripper")
        backend.connect()
        assert backend.read_joints_deg().shape == (6,)

    def test_refuses_io_before_connecting(self) -> None:
        backend = MockFollower()
        with pytest.raises(RuntimeError):
            backend.read_joints_deg()
        with pytest.raises(RuntimeError):
            backend.write_joints_deg(np.zeros(6))

    def test_rejects_a_wrong_length_command(self) -> None:
        backend = MockFollower()
        backend.connect()
        with pytest.raises(ValueError):
            backend.write_joints_deg(np.zeros(5))

    def test_lags_behind_the_command_rather_than_snapping(self) -> None:
        backend = MockFollower(read_noise_deg=0.0, tracking_gain=0.35)
        backend.connect()
        start = backend.true_joints_deg()
        target = start + 10.0

        backend.write_joints_deg(target)
        after_one = backend.true_joints_deg()

        assert np.all(after_one > start), "must move toward the target"
        assert np.all(after_one < target), "must not arrive in a single step"

        for _ in range(60):
            backend.write_joints_deg(target)
        assert np.allclose(backend.true_joints_deg(), target, atol=1e-3), "must converge"

    def test_never_exceeds_joint_limits(self) -> None:
        backend = MockFollower(read_noise_deg=0.0)
        backend.connect()
        lo, hi = ArmConfig().limits_deg_array()

        for _ in range(50):
            backend.write_joints_deg(np.full(6, 1e4))
        assert np.all(backend.true_joints_deg()[:5] <= hi + 1e-9)

        for _ in range(50):
            backend.write_joints_deg(np.full(6, -1e4))
        assert np.all(backend.true_joints_deg()[:5] >= lo - 1e-9)


class TestTeleopLoop:
    def test_runs_a_full_sweep_without_hardware(self, ik: TaskIK5D) -> None:
        source = ScriptedSource.from_waveform(n_steps=200)
        loop = make_loop(source, ik)

        stats = loop.run(realtime=False)
        summary = stats.summary()

        assert summary["steps"] == 200
        assert loop.backend.write_count == 200

    def test_tracks_default_5d_waveform_tightly(self, ik: TaskIK5D) -> None:
        source = ScriptedSource.from_waveform(n_steps=300, amplitude=0.2)
        loop = make_loop(source, ik, read_noise_deg=0.0)

        summary = loop.run(realtime=False).summary()

        assert summary["ik_pos_err_p95_mm"] < 0.05, summary
        assert summary["ik_pitch_err_p95_deg"] < 0.05, summary
        assert summary["ik_solver_converged_frac"] > 0.97, summary

    def test_aggressive_imu_sweep_reports_atlas_and_rate_walls(self, ik: TaskIK5D) -> None:
        source = ScriptedSource.from_waveform(n_steps=300, amplitude=1.0, rotation_amplitude_rad=0.35)
        loop = make_loop(source, ik, read_noise_deg=0.0)

        summary = loop.run(realtime=False).summary()

        assert summary["atlas_pitch_clamped_frac"] > 0.0, summary
        assert summary["joint_rate_clamped_frac"] > 0.0, summary
        assert summary["ik_pos_err_max_mm"] > 10.0, summary

    def test_never_commands_out_of_limit_joints(self, ik: TaskIK5D) -> None:
        source = ScriptedSource.from_waveform(n_steps=300, amplitude=1.0, rotation_amplitude_rad=0.35)
        loop = make_loop(source, ik)
        lo, hi = ArmConfig().limits_deg_array()

        stats = loop.run(realtime=False)

        for record in stats.records:
            assert np.all(record.commanded_joints_deg >= lo - 1e-9), record.index
            assert np.all(record.commanded_joints_deg <= hi + 1e-9), record.index

    def test_never_exceeds_the_per_step_joint_velocity_cap(self, ik: TaskIK5D) -> None:
        source = ScriptedSource.from_waveform(n_steps=300, amplitude=1.0, rotation_amplitude_rad=0.35)
        loop = make_loop(source, ik)
        cap = loop.config.teleop.max_joint_step_deg

        stats = loop.run(realtime=False)
        commands = np.array([r.commanded_joints_deg for r in stats.records])
        steps = np.abs(np.diff(commands, axis=0))

        assert steps.max() <= cap + 1e-6, (
            f"largest command-to-command step {steps.max():.4f} deg exceeds the {cap:g} deg cap"
        )

    def test_starts_from_the_arm_pose_not_the_configured_home(self, ik: TaskIK5D) -> None:
        start_joints = np.array([20.0, 75.0, -75.0, 25.0, 15.0, 45.0])
        expected = ik.task_pose(start_joints[:5]).pose
        limits = SoSnakeConfig().limits
        assert np.all(expected.position >= np.asarray(limits.pos_min_m)), "test pose must start in the box"
        assert np.all(expected.position <= np.asarray(limits.pos_max_m)), "test pose must start in the box"

        source = ScriptedSource(samples=[sample()])
        backend = MockFollower(initial_joints_deg=start_joints, read_noise_deg=0.0)
        loop = TeleopLoop(source, backend, SoSnakeConfig(), ik=ik)

        backend.connect()
        loop.sync_target_to_arm()

        assert np.allclose(loop.tracker.pose.as_array(), expected.as_array(), atol=1e-9)

    def test_an_arm_starting_outside_the_workspace_is_pulled_back_in(self, ik: TaskIK5D) -> None:
        start_joints = np.array([0.0, 100.27, -90.0, -37.25, 0.0, 51.56])
        outside = ik.task_pose(start_joints[:5]).pose.position
        limits = SoSnakeConfig().limits
        assert np.any(outside > np.asarray(limits.pos_max_m)), "test premise: starts outside"

        source = ScriptedSource(samples=[sample()])
        backend = MockFollower(initial_joints_deg=start_joints, read_noise_deg=0.0)
        loop = TeleopLoop(source, backend, SoSnakeConfig(), ik=ik)

        backend.connect()
        loop.sync_target_to_arm()

        assert np.all(loop.tracker.pose.position >= np.asarray(limits.pos_min_m) - 1e-12)
        assert np.all(loop.tracker.pose.position <= np.asarray(limits.pos_max_m) + 1e-12)

    def test_stops_when_the_source_says_stop(self, ik: TaskIK5D) -> None:
        samples = [sample(t=float(i)) for i in range(5)]
        samples.append(sample(t=5.0, events=frozenset({"stop"})))
        loop = make_loop(ScriptedSource(samples=samples), ik)

        stats = loop.run(max_steps=100, realtime=False)

        assert stats.summary()["steps"] == 5

    def test_reverses_immediately_after_holding_against_workspace_wall(self, ik: TaskIK5D) -> None:
        samples = [sample(left=(0.0, 1.0)) for _ in range(60)]
        samples.extend(sample(left=(0.0, 1.0)) for _ in range(20))
        samples.append(sample(left=(0.0, -1.0)))
        loop = make_loop(ScriptedSource(samples=samples), ik, read_noise_deg=0.0)

        records = loop.run(realtime=False).records
        before_reverse = records[-2]
        reverse = records[-1]
        x_max = loop.config.limits.pos_max_m[0]

        assert before_reverse.workspace_clamped
        assert before_reverse.task_target[0] == pytest.approx(x_max)
        assert reverse.task_target[0] < before_reverse.task_target[0]
        assert not reverse.workspace_clamped

    def test_holds_position_while_the_clutch_is_released(self, ik: TaskIK5D) -> None:
        samples = [sample(left=(1.0, 1.0), right=(0.0, 1.0), clutch=False) for _ in range(20)]
        source = ScriptedSource(samples=samples)
        loop = make_loop(source, ik, read_noise_deg=0.0)

        loop.backend.connect()
        loop.sync_target_to_arm()
        held = loop.tracker.pose.as_array()

        loop.run(realtime=False)

        assert np.allclose(loop.tracker.pose.as_array(), held, atol=1e-9)
        assert all(not record.clutch_engaged for record in loop.stats.records)

    def test_gripper_command_spans_the_configured_range(self, ik: TaskIK5D) -> None:
        teleop = SoSnakeConfig().teleop
        samples = [sample(gripper=g) for g in (1.0, 0.0, 1.0, 0.0)]
        loop = make_loop(ScriptedSource(samples=samples), ik)

        stats = loop.run(realtime=False)
        grips = [r.gripper_cmd_deg for r in stats.records]

        assert grips[0] == pytest.approx(teleop.gripper_open_deg)
        assert grips[1] == pytest.approx(teleop.gripper_closed_deg)

    def test_holds_previous_command_when_post_rate_pose_falls_below_workspace_floor(
        self, ik: TaskIK5D, monkeypatch
    ) -> None:
        source = ScriptedSource(samples=[sample(clutch=True)])
        loop = make_loop(source, ik, read_noise_deg=0.0)
        unsafe_joints = np.array([0.0, 0.0, -180.0, -143.24, 0.0])
        unsafe_pose = ik.task_pose(unsafe_joints).pose
        assert unsafe_pose.z < SoSnakeConfig().limits.pos_min_m[2], "test command must be below the floor"

        def unsafe_solve(seed, target, *, rate_reference_deg=None, rate_limit=True):
            del seed, target, rate_reference_deg, rate_limit
            T = ik.forward(unsafe_joints)
            return IK5DResult(
                joints_deg=unsafe_joints.copy(),
                raw_joints_deg=unsafe_joints.copy(),
                achieved=unsafe_pose,
                achieved_yaw_rad=0.0,
                achieved_yaw_residual_rad=0.0,
                achieved_pose_world=T,
                position_error_m=0.0,
                pitch_error_rad=0.0,
                roll_error_rad=0.0,
                iterations=1,
                converged=False,
                solver_converged=False,
                min_singular_value=0.0,
                damping=0.0,
                limit_clamped=np.zeros(5, dtype=bool),
                rate_clamped=np.ones(5, dtype=bool),
            )

        monkeypatch.setattr(loop.ik, "solve", unsafe_solve)
        record = loop.run(realtime=False).records[0]

        assert record.command_safety_held
        assert record.command_safety_reason == "post_rate_achieved_z_below_workspace_floor"
        assert record.achieved_position[2] >= loop.config.limits.pos_min_m[2]
        assert np.allclose(record.commanded_joints_deg, loop.config.teleop.home_joints_deg, atol=0.2)

    def test_holds_previous_command_when_candidate_mesh_is_too_low(self, ik: TaskIK5D) -> None:
        source = ScriptedSource(samples=[sample(clutch=True, left=(0.0, 1.0))])
        loop = make_loop(source, ik, read_noise_deg=0.0)

        def low_mesh(_target_deg):
            return 0.010, "jaw"

        loop.backend.command_robot_mesh_min_z_deg = low_mesh
        record = loop.run(realtime=False).records[0]

        assert record.command_safety_held
        assert record.command_safety_reason == "post_rate_robot_mesh_below_clearance:jaw:0.0100"
        assert record.robot_mesh_min_z_m == pytest.approx(0.010)
        assert record.robot_mesh_min_body == "jaw"
        assert np.allclose(record.commanded_joints_deg, loop.config.teleop.home_joints_deg, atol=0.2)

    def test_stick_translation_gain_is_five_times_xyz(self, ik: TaskIK5D) -> None:
        left = sample(left=(-1.0, -1.0), right=(0.0, -1.0))
        loop = make_loop(ScriptedSource(samples=[left]), ik, read_noise_deg=0.0)

        record = loop.run(realtime=False).records[0]
        step = np.asarray(loop.config.teleop.translation_step_m)
        gain = loop.config.teleop.stick_translation_gain

        assert record.task_delta[:3] == pytest.approx(np.array([-gain * step[0], gain * step[1], -gain * step[2]]))

    def test_records_all_three_action_streams(self, ik: TaskIK5D) -> None:
        loop = make_loop(ScriptedSource(samples=[sample(left=(0.0, 1.0), gripper=0.25)]), ik)

        record = loop.run(realtime=False).records[0]

        assert "action.raw.sticks" in record.raw
        assert record.task_target.shape == (5,)
        assert record.task_delta.shape == (5,)
        assert record.commanded_joints_deg.shape == (5,)
        assert record.measured_joints_deg.shape == (5,)

    def test_is_deterministic_given_the_same_script(self, ik: TaskIK5D) -> None:
        def run() -> np.ndarray:
            loop = make_loop(ScriptedSource.from_waveform(n_steps=120), ik, seed=7)
            stats = loop.run(realtime=False)
            return np.array([r.commanded_joints_deg for r in stats.records])

        assert np.allclose(run(), run())
