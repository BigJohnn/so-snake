"""move_to_joints: a rate-limited, monotone joint-space move that converges."""

import numpy as np
import pytest

from so_snake.config import ARM_JOINTS, GRIPPER_JOINT, SoSnakeConfig
from so_snake.m4_execution import MockFollower, move_to_joints


def _backend(start):
    b = MockFollower(arm=SoSnakeConfig().arm, tracking_gain=0.35, read_noise_deg=0.0,
                     initial_joints_deg=np.asarray(start, float))
    b.connect()
    return b


def test_converges_to_target():
    target = np.array([8.4, 75.8, -46.0, 42.2, 3.6, 89.6])
    b = _backend([6.9, 23.5, 3.1, 33.4, -6.2, 50.0])
    reached = move_to_joints(b, target, step_deg=1.5, tol_deg=1.0, hz=30.0, sleep=lambda _: None)
    assert reached
    assert np.abs(b.true_joints_deg() - target).max() <= 1.0


def test_command_steps_are_rate_limited():
    target = np.array([100.0, 100.0, -100.0, 60.0, 100.0, 100.0])
    b = _backend([0.0, 70.0, -70.0, 30.0, 0.0, 50.0])
    steps = []
    orig = b.write_joints_deg

    def spy(cmd):
        steps.append(np.asarray(cmd, float) - b.true_joints_deg())
        orig(cmd)

    b.write_joints_deg = spy
    move_to_joints(b, target, step_deg=2.0, tol_deg=1.0, hz=30.0, sleep=lambda _: None)
    # every commanded step is within step_deg of the present position (+ tiny slack)
    assert max(np.abs(s).max() for s in steps) <= 2.0 + 1e-9


def test_returns_false_if_unreachable_in_budget():
    # A target the (lagging) servo cannot reach within a tiny iteration budget.
    b = _backend([0.0, 70.0, -70.0, 30.0, 0.0, 50.0])
    reached = move_to_joints(b, np.array([114.0, 200.0, -180.0, 60.0, 179.0, 100.0]),
                             step_deg=1.0, tol_deg=0.01, hz=30.0, max_extra_steps=2, sleep=lambda _: None)
    assert reached is False
