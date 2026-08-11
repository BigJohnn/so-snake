"""move_to_joints: a rate-limited, monotone joint-space move that converges."""

import inspect

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
    outcome = move_to_joints(b, np.array([114.0, 200.0, -180.0, 60.0, 179.0, 100.0]),
                             step_deg=1.0, tol_deg=0.01, hz=30.0, max_extra_steps=2, sleep=lambda _: None)
    assert not outcome
    assert outcome.reached is False


def test_the_outcome_says_which_joints_are_short_and_by_how_much():
    """A bare False is what made a real replay abort with nothing to go on.

    The caller has to be able to tell "the shoulder settled 2.7 deg out, which
    is all this servo can do" from "something is holding the arm".
    """
    b = _backend([0.0, 70.0, -70.0, 30.0, 0.0, 50.0])
    target = np.array([60.0, 70.0, -70.0, 30.0, 0.0, 50.0])  # only shoulder_pan moves
    outcome = move_to_joints(b, target, step_deg=1.0, tol_deg=0.01, hz=30.0,
                             max_extra_steps=2, sleep=lambda _: None)

    assert outcome.joint_names[0] == ARM_JOINTS[0]
    worst_joint, worst_deg = outcome.worst()[0]
    assert worst_joint == ARM_JOINTS[0]
    assert worst_deg == pytest.approx(outcome.max_residual_deg)
    assert ARM_JOINTS[0] in outcome.describe()
    # The joints that never had to move are not reported as residuals.
    assert outcome.residual_deg[1:].max() < 0.01


def test_a_jammed_joint_ends_the_move_instead_of_grinding_at_it():
    """Real symptom: the arm settles a couple of degrees out and stops moving.

    Before this, the move spent its whole iteration budget -- seven seconds of
    pushing at something it could not reach -- and then reported a plain False.
    """
    b = _backend([0.0, 70.0, -70.0, 30.0, 0.0, 50.0])
    frozen = b.true_joints_deg().copy()
    # A servo that accepts commands and does not move: an obstruction, a joint
    # limit, or a lead too small to overcome gravity all look like this.
    b.write_joints_deg = lambda cmd: None
    b.read_joints_deg = lambda: frozen.copy()

    outcome = move_to_joints(b, frozen + 20.0, step_deg=6.0, tol_deg=1.0, hz=30.0,
                             sleep=lambda _: None)

    assert not outcome.reached and outcome.stalled
    assert outcome.max_residual_deg == pytest.approx(20.0)
    # A second of no progress at 30 Hz, not the full 200-step budget.
    assert outcome.steps < 60


def test_an_interrupted_move_says_so_rather_than_looking_stuck():
    """Stop is not the same failure as an obstruction, and must not read as one."""
    b = _backend([0.0, 70.0, -70.0, 30.0, 0.0, 50.0])
    calls = {"n": 0}

    def should_continue() -> bool:
        calls["n"] += 1
        return calls["n"] < 5

    outcome = move_to_joints(b, np.array([60.0, 70.0, -70.0, 30.0, 0.0, 50.0]),
                             step_deg=6.0, tol_deg=1.0, hz=30.0,
                             should_continue=should_continue, sleep=lambda _: None)
    assert not outcome.reached
    assert outcome.interrupted and not outcome.stalled


def test_the_default_tolerance_is_what_the_servo_can_actually_hold():
    """1.0 deg was below the measured standing offset, so nothing ever arrived.

    Measured on the bench from the recorded takes: shoulder_pan sits ~2.7 deg
    off its command at rest (lerobot halves the servo's P gain), so a homing
    move or a replay approach asking for 1.0 deg could never report success.
    """
    tol = SoSnakeConfig().teleop.joint_settle_tol_deg
    assert tol >= 2.7
    assert inspect.signature(move_to_joints).parameters["tol_deg"].default == tol
