"""The GUI session manager's one invariant: one thing drives the arm at a time."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from so_snake.config import SoSnakeConfig
from so_snake.data import EpisodeStore, ReplayConfig
from so_snake.gui.session import LockedBackend, SessionManager
from so_snake.m4_execution import MockFollower
from so_snake.rig import RigSpec
from so_snake.start_pose import JOINT_ORDER


@pytest.fixture
def manager(tmp_path) -> SessionManager:
    # The start pose goes to a temporary file: capturing one is a write, and the
    # bench's own assets/so100_start_pose.json is not a test fixture.
    session = SessionManager(SoSnakeConfig(), tmp_path, start_pose_path=tmp_path / "start_pose.json")
    yield session
    session.shutdown()


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


MOCK = RigSpec(backend="mock", source="scripted", scripted_steps=600, scripted_loop=True)


# ------------------------------------------------------------ locked backend


def test_locked_backend_does_not_invent_a_clearance_probe():
    """The mock has no geometry, and must keep saying so.

    `TeleopLoop` decides whether to run the mesh clearance check by asking
    `getattr(backend, "command_robot_mesh_min_z_deg", None)`. A wrapper that
    answered for backends without one would turn every mock run into a crash.
    """
    wrapped = LockedBackend(MockFollower(), threading.Lock())
    assert getattr(wrapped, "command_robot_mesh_min_z_deg", None) is None
    assert wrapped.joint_names == MockFollower().joint_names


# ------------------------------------------------------------------- session


def test_session_runs_and_stops_cleanly(manager):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.status()["steps"] > 5)

    status = manager.status()
    assert status["mode"] == "teleop"
    assert status["latest"]["loop_hz"] > 0

    manager.stop()
    assert manager.mode == "idle"
    # The spec goes with the session: showing a backend next to "idle" reads as
    # loaded and ready when nothing is connected.
    assert manager.status()["spec"] is None


def test_a_second_session_is_refused_while_one_is_running(manager):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    with pytest.raises(RuntimeError, match="busy"):
        manager.start_session(MOCK)
    manager.stop()


def test_replay_is_refused_while_teleoperating(manager):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    with pytest.raises(RuntimeError, match="busy"):
        manager.start_replay("whatever", MOCK, ReplayConfig())
    manager.stop()


def test_homing_is_refused_while_teleoperating(manager):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    with pytest.raises(RuntimeError, match="busy"):
        manager.start_homing(MOCK)
    manager.stop()


# ----------------------------------------------------------------- recording


def test_recording_requires_a_session(manager):
    with pytest.raises(RuntimeError, match="teleop session"):
        manager.start_recording(task="pick the cube")


def test_record_start_stop_writes_an_episode(manager, tmp_path):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")

    manager.start_recording(name="take 1", task="pick the cube")
    assert wait_until(lambda: manager.status()["recording"]["steps"] > 5)
    manager.stop_recording(keep=True)
    manager.stop()

    episodes = EpisodeStore(tmp_path).list_meta()
    assert len(episodes) == 1
    assert episodes[0].task == "pick the cube"
    assert episodes[0].n_steps > 5


def test_stopping_a_session_mid_recording_keeps_the_take_and_flags_it(manager, tmp_path):
    """A crash or a fumbled stop must not silently drop the demonstration."""
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    manager.start_recording(task="interrupted")
    assert wait_until(lambda: manager.status()["recording"]["steps"] > 5)

    manager.stop()

    episodes = EpisodeStore(tmp_path).list_meta()
    assert len(episodes) == 1
    assert "session ended" in episodes[0].aborted_reason


def test_discarded_take_writes_nothing(manager, tmp_path):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    manager.start_recording(task="fumbled")
    assert wait_until(lambda: manager.status()["recording"]["steps"] > 5)
    manager.stop_recording(keep=False)
    manager.stop()

    assert EpisodeStore(tmp_path).list_meta() == []


# --------------------------------------------------------------- held at home


def test_homing_ends_holding_the_pose_rather_than_releasing_it(manager):
    """Torque stays on after a homing move -- that is the whole point of it.

    A real arm that is released at the end of a homing move does not stay at the
    home pose: gravity has the elbow before the operator can do anything with
    it, so the pose they asked for is gone by the time they can teleoperate from
    it.
    """
    manager.start_homing(MOCK)
    assert wait_until(lambda: manager.mode == "held")

    status = manager.status()
    assert status["connected"] is True
    # The spec survives too: the UI shows which arm is standing there energized.
    assert status["spec"]["backend"] == "mock"


def test_teleop_started_from_held_adopts_the_same_connected_backend(manager):
    """No second backend, and no gap in torque, between homing and teleop.

    Identity is the assertion that matters: on hardware a second backend means a
    second handle on the same serial port, and getting one would require
    disconnecting the first -- which is exactly the drop in torque that holding
    the pose exists to avoid.
    """
    manager.start_homing(MOCK)
    assert wait_until(lambda: manager.mode == "held")
    held_backend = manager.status()["connected"]
    inner = manager._backend.inner

    manager.start_session(MOCK)
    assert wait_until(lambda: manager.status()["steps"] > 3)
    assert manager._backend.inner is inner
    assert held_backend is True and inner.is_connected

    manager.stop()
    assert manager.mode == "idle"


def test_stop_is_the_only_thing_that_releases_a_held_arm(manager):
    manager.start_homing(MOCK)
    assert wait_until(lambda: manager.mode == "held")
    inner = manager._backend.inner

    manager.stop()
    assert manager.mode == "idle"
    assert not inner.is_connected
    assert manager.status()["spec"] is None


def test_a_held_arm_refuses_a_session_on_a_different_arm(manager):
    """Silently switching arms would mean the held one is left energized."""
    manager.start_homing(MOCK)
    assert wait_until(lambda: manager.mode == "held")

    other = RigSpec(backend="mock", source="scripted", port="/dev/ttyUSB9")
    with pytest.raises(RuntimeError, match="held at home"):
        manager.start_session(other)

    assert manager.mode == "held"
    manager.stop()


def test_recording_is_refused_while_the_arm_is_merely_held(manager):
    manager.start_homing(MOCK)
    assert wait_until(lambda: manager.mode == "held")
    with pytest.raises(RuntimeError, match="teleop session"):
        manager.start_recording(task="too early")
    manager.stop()


# ------------------------------------------------------------- takes in a batch


def test_a_fixed_length_take_stops_itself_and_then_homes(manager, tmp_path):
    """The length is the episode's, not the operator's reaction time.

    Episodes an operator ends by hand vary by seconds, and a model trained on
    them sees that variance as if it said something about the task.
    """
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")

    manager.start_recording(task="fixed length", steps=20, target_count=3)
    # `done_count` rather than the recording flag: it is incremented after the
    # episode is on disk, so this is not a race with the writer.
    assert wait_until(lambda: manager.status()["takes"]["done_count"] == 1, timeout=15.0)

    episodes = EpisodeStore(tmp_path).list_meta()
    assert len(episodes) == 1
    assert episodes[0].n_steps == 20

    takes = manager.status()["takes"]
    assert (takes["done_count"], takes["target_count"], takes["steps_per_take"]) == (1, 3, 20)

    # ...and the arm walks home on its own, then waits: teleop is still live and
    # the next take is one button press away, with no take running.
    assert wait_until(lambda: manager.mode == "teleop" and not manager.status()["recording"]["recording"])
    manager.stop()


def test_the_next_take_can_start_after_the_automatic_homing(manager, tmp_path):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")

    manager.start_recording(task="first", steps=10, target_count=2)
    assert wait_until(lambda: manager.status()["takes"]["done_count"] == 1, timeout=15.0)
    # Homing runs on the worker; the second take may only start once it is done.
    assert wait_until(lambda: manager.mode == "teleop", timeout=15.0)

    manager.start_recording(task="second", steps=10, target_count=2)
    assert wait_until(lambda: manager.status()["takes"]["done_count"] == 2, timeout=15.0)
    manager.stop()

    tasks = sorted(meta.task for meta in EpisodeStore(tmp_path).list_meta())
    assert tasks == ["first", "second"]


def test_an_unbounded_take_still_runs_until_the_operator_stops_it(manager, tmp_path):
    """steps=0 is the old behaviour, and stays available."""
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    manager.start_recording(task="manual", steps=0)
    assert wait_until(lambda: manager.status()["recording"]["steps"] > 25)
    assert manager.status()["recording"]["recording"] is True
    manager.stop_recording(keep=True)
    manager.stop()
    assert EpisodeStore(tmp_path).list_meta()[0].n_steps > 25


# ------------------------------------------------------- the recorded start pose


def test_the_start_pose_is_recorded_from_the_arm_and_becomes_the_homing_target(manager):
    """Fly the arm somewhere, press the button, and homing goes there from then on."""
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.status()["steps"] > 5)

    # Where the arm actually is right now -- the scripted waveform has moved it
    # off the configured home by this point.
    measured = manager._backend.read_joints_deg()
    manager.capture_start_pose()

    described = manager.status()["start_pose"]
    assert described["source"] == "file"
    recorded = np.array([described["joints_deg"][name] for name in JOINT_ORDER])
    assert np.allclose(recorded, measured, atol=0.5)
    # And that is what the next homing move aims at, not TeleopConfig's home.
    assert np.allclose(manager._home_target(), recorded, atol=0.01)
    manager.stop()


def test_homing_falls_back_to_the_configured_home_without_a_recorded_pose(manager):
    target = manager._home_target()
    expected = np.array([*manager.config.teleop.home_joints_deg, 0.0])
    assert np.allclose(target[:-1], expected[:-1])


def test_an_unusable_start_pose_file_does_not_block_homing(manager):
    """Refusing to park an energized arm because a JSON file is broken is worse."""
    manager.start_pose_path.write_text("{ not json")

    target = manager._home_target()
    assert np.allclose(target[:-1], np.array(manager.config.teleop.home_joints_deg))
    assert any("homing to the configured home pose" in e.message for e in manager._events)


def test_capturing_a_start_pose_needs_a_live_arm(manager):
    with pytest.raises(RuntimeError, match="no live arm"):
        manager.capture_start_pose()


def test_a_start_pose_can_be_captured_from_a_held_arm(manager):
    """The other useful moment: home, nudge nothing, and pin that pose down."""
    manager.start_homing(MOCK)
    assert wait_until(lambda: manager.mode == "held")
    manager.capture_start_pose()
    assert manager.status()["start_pose"]["source"] == "file"
    manager.stop()


# ------------------------------------------------- keeping or discarding a take


def test_a_take_that_ends_by_itself_waits_for_a_verdict(manager, tmp_path):
    """Running out of frames is not somebody deciding the take was good."""
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    manager.start_recording(task="judge me", steps=15)
    assert wait_until(lambda: manager.status()["takes"]["done_count"] == 1, timeout=15.0)

    last = manager.status()["last_take"]
    assert last["pending"] is True and last["n_steps"] == 15
    # Written already: a take held in memory until approved is one a crash loses.
    assert (tmp_path / last["id"]).is_dir()
    manager.stop()


def test_discarding_the_last_take_deletes_it_and_uncounts_it(manager, tmp_path):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    manager.start_recording(task="not good", steps=15, target_count=5)
    assert wait_until(lambda: manager.status()["takes"]["done_count"] == 1, timeout=15.0)
    episode_id = manager.status()["last_take"]["id"]

    manager.decide_last_take(keep=False)

    assert not (tmp_path / episode_id).exists()
    assert EpisodeStore(tmp_path).list_meta() == []
    # The batch count is a count of usable episodes, so a discard takes one back.
    assert manager.status()["takes"]["done_count"] == 0
    assert manager.status()["last_take"]["pending"] is False
    manager.stop()


def test_keeping_the_last_take_only_dismisses_the_prompt(manager, tmp_path):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    manager.start_recording(task="good one", steps=15)
    assert wait_until(lambda: manager.status()["takes"]["done_count"] == 1, timeout=15.0)

    manager.decide_last_take(keep=True)
    assert manager.status()["last_take"]["pending"] is False
    assert manager.status()["takes"]["done_count"] == 1
    assert len(EpisodeStore(tmp_path).list_meta()) == 1
    manager.stop()


def test_a_manually_saved_take_is_not_asked_about_again(manager):
    """The operator pressed save; asking "did you mean it" is noise."""
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    manager.start_recording(task="manual", steps=0)
    assert wait_until(lambda: manager.status()["recording"]["steps"] > 5)
    manager.stop_recording(keep=True)

    assert manager.status()["last_take"]["pending"] is False
    with pytest.raises(RuntimeError, match="waiting for a verdict"):
        manager.decide_last_take(keep=False)
    manager.stop()


def test_starting_the_next_take_counts_as_keeping_the_last(manager, tmp_path):
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    manager.start_recording(task="first", steps=15)
    assert wait_until(lambda: manager.status()["takes"]["done_count"] == 1, timeout=15.0)
    assert wait_until(lambda: manager.mode == "teleop", timeout=15.0)  # auto-home done

    manager.start_recording(task="second", steps=15)
    assert manager.status()["last_take"]["pending"] is False
    assert wait_until(lambda: manager.status()["takes"]["done_count"] == 2, timeout=15.0)
    manager.stop()

    assert len(EpisodeStore(tmp_path).list_meta()) == 2


def test_preview_is_none_without_a_simulator(manager):
    """The mock and the real arm have nothing to render, and must say so.

    Returning None here is what makes the gateway serve a placeholder rather
    than a broken image or a 500.
    """
    assert manager.preview_frame("third_person", 64, 48) is None
    manager.start_session(MOCK)
    assert wait_until(lambda: manager.mode == "teleop")
    assert manager.preview_frame("third_person", 64, 48) is None
    manager.stop()
