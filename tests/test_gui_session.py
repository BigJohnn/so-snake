"""The GUI session manager's one invariant: one thing drives the arm at a time."""

from __future__ import annotations

import threading
import time

import pytest

from so_snake.config import SoSnakeConfig
from so_snake.data import EpisodeStore, ReplayConfig
from so_snake.gui.session import LockedBackend, SessionManager
from so_snake.m4_execution import MockFollower
from so_snake.rig import RigSpec


@pytest.fixture
def manager(tmp_path) -> SessionManager:
    session = SessionManager(SoSnakeConfig(), tmp_path)
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
