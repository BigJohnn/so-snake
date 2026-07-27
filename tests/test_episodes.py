"""Recording and replay: the round trip, the safety clamps, and the library."""

from __future__ import annotations

import json

import numpy as np
import pytest

from so_snake.config import SoSnakeConfig
from so_snake.data import (
    COLUMN_NAMES,
    Episode,
    EpisodeMeta,
    EpisodeRecorder,
    EpisodeReplayer,
    EpisodeStore,
    ReplayConfig,
    inspect_episode,
)
from so_snake.data.episode import META_NAME
from so_snake.m4_execution import MockFollower
from so_snake.teleop import ScriptedSource, TeleopLoop


@pytest.fixture(scope="module")
def config() -> SoSnakeConfig:
    return SoSnakeConfig()


def record(root, config, *, steps: int = 80, name: str = "", task: str = "", keep: bool = True):
    """Drive the real loop into a real episode. Nothing here is a stub."""
    backend = MockFollower()
    loop = TeleopLoop(ScriptedSource.from_waveform(steps, amplitude=0.2), backend, config)
    recorder = EpisodeRecorder(
        root, config=config, backend="mock", source="scripted", joint_names=backend.joint_names
    )
    recorder.start(name=name, task=task)
    loop.run(max_steps=steps, realtime=False, on_step=recorder.append)
    return recorder.stop(keep=keep)


# --------------------------------------------------------------------- format


def test_recording_writes_every_column(tmp_path, config):
    meta = record(tmp_path, config, steps=60, task="pick the cube")
    assert meta is not None
    assert meta.n_steps == 60
    assert meta.task == "pick the cube"

    episode = EpisodeStore(tmp_path).load(meta.id)
    assert set(episode.frames) == set(COLUMN_NAMES)
    for column in COLUMN_NAMES:
        assert len(episode.frames[column]) == 60, column


def test_meta_snapshots_the_config_that_shaped_the_episode(tmp_path, config):
    meta = record(tmp_path, config, steps=20)
    # An episode recorded under a workspace box that later changes has to stay
    # interpretable, so the box travels with it rather than being looked up.
    assert meta.config["limits"]["pos_min_m"] == list(config.limits.pos_min_m)
    assert meta.config["teleop"]["max_joint_step_deg"] == config.teleop.max_joint_step_deg
    assert meta.config["arm"]["urdf_path"] == str(config.arm.urdf_path)


def test_discarded_take_leaves_nothing_on_disk(tmp_path, config):
    assert record(tmp_path, config, steps=20, keep=False) is None
    assert EpisodeStore(tmp_path).list_meta() == []


def test_step_cap_stops_buffering_and_flags_the_episode(tmp_path, config):
    backend = MockFollower()
    loop = TeleopLoop(ScriptedSource.from_waveform(40, amplitude=0.2), backend, config)
    recorder = EpisodeRecorder(
        tmp_path, config=config, backend="mock", source="scripted",
        joint_names=backend.joint_names, max_steps=10,
    )
    recorder.start()
    loop.run(max_steps=40, realtime=False, on_step=recorder.append)
    meta = recorder.stop()

    assert meta is not None
    assert meta.n_steps == 10
    assert "step cap" in meta.aborted_reason


# ---------------------------------------------------------------------- store


def test_store_skips_directories_that_are_not_episodes(tmp_path, config):
    good = record(tmp_path, config, steps=20)
    (tmp_path / "not_an_episode").mkdir()
    (tmp_path / "corrupt").mkdir()
    (tmp_path / "corrupt" / META_NAME).write_text("{ this is not json", encoding="utf-8")

    # One bad directory must not cost the operator the rest of the session.
    assert [m.id for m in EpisodeStore(tmp_path).list_meta()] == [good.id]


def test_store_refuses_ids_that_escape_the_root(tmp_path):
    store = EpisodeStore(tmp_path)
    for bad in ("../etc", "..", "a/b", ""):
        with pytest.raises(ValueError):
            store.path_of(bad)


def test_annotate_edits_labels_only(tmp_path, config):
    meta = record(tmp_path, config, steps=20, task="old")
    store = EpisodeStore(tmp_path)
    store.annotate(meta.id, task="new", notes="cube was off-centre")

    payload = json.loads((tmp_path / meta.id / META_NAME).read_text(encoding="utf-8"))
    assert payload["task"] == "new"
    assert payload["notes"] == "cube was off-centre"
    assert payload["n_steps"] == meta.n_steps
    assert payload["config"] == meta.config


def test_delete_removes_the_directory(tmp_path, config):
    meta = record(tmp_path, config, steps=20)
    store = EpisodeStore(tmp_path)
    assert store.delete(meta.id) is True
    assert not (tmp_path / meta.id).exists()
    assert store.delete(meta.id) is False


# --------------------------------------------------------------------- replay


@pytest.mark.parametrize("mode", ["joint", "task"])
def test_replay_reproduces_the_recorded_commands(tmp_path, config, mode):
    meta = record(tmp_path, config, steps=80)
    episode = EpisodeStore(tmp_path).load(meta.id)

    report = EpisodeReplayer(
        episode, MockFollower(), config, ReplayConfig(mode=mode, realtime=False)
    ).run()

    assert report.completed
    assert report.n_steps == meta.n_steps
    # Joint mode is a replay of the exact commands, so any deviation is a clamp
    # firing. Task mode re-solves and must land on the same branch, which is the
    # property that makes it usable as a regression test at all.
    assert report.summary()["command_deviation_max_deg"] < 1e-6


def test_replay_rate_cap_binds_when_played_faster_than_recorded(tmp_path, config):
    meta = record(tmp_path, config, steps=80)
    episode = EpisodeStore(tmp_path).load(meta.id)
    commanded = episode.commanded_joints_deg
    peak_step = float(np.abs(np.diff(commanded, axis=0)).max())

    # Cap the arm at a tenth of the rate the recording used, so playback cannot
    # help but be clamped -- and check that it is the commands that give way,
    # not the cap.
    cap = peak_step * config.teleop.control_hz / 10.0
    report = EpisodeReplayer(
        episode,
        MockFollower(),
        config,
        ReplayConfig(mode="joint", realtime=False, max_joint_rate_deg_s=cap),
    ).run()

    assert report.completed
    assert report.summary()["rate_clamped_frac"] > 0.0
    steps = np.abs(np.diff([s.commanded_joints_deg for s in report.steps], axis=0))
    assert steps.max() <= cap / config.teleop.control_hz + 1e-6


def test_replay_clamps_to_current_joint_limits(tmp_path, config):
    meta = record(tmp_path, config, steps=40)
    episode = EpisodeStore(tmp_path).load(meta.id)

    # An episode recorded before the limits were tightened must be clamped to
    # the arm's limits, not the recording's -- the limits describe the hardware.
    tightened = SoSnakeConfig(
        arm=type(config.arm)(
            joint_limits_deg={**config.arm.joint_limits_deg, "shoulder_pan": (-1.0, 1.0)}
        )
    )
    report = EpisodeReplayer(
        episode, MockFollower(arm=tightened.arm), tightened,
        ReplayConfig(mode="joint", realtime=False),
    ).run()

    pan = np.array([s.commanded_joints_deg[0] for s in report.steps])
    assert pan.min() >= -1.0 - 1e-6
    assert pan.max() <= 1.0 + 1e-6


def test_replay_approaches_the_first_frame_before_playing(tmp_path, config):
    meta = record(tmp_path, config, steps=40)
    episode = EpisodeStore(tmp_path).load(meta.id)

    # Start the arm somewhere else entirely. Playback must walk it in rather
    # than sending frame 0 from wherever it happens to be.
    away = np.array([60.0, 120.0, -30.0, 0.0, 90.0, 40.0])
    backend = MockFollower(initial_joints_deg=away)
    report = EpisodeReplayer(
        episode, backend, config, ReplayConfig(mode="joint", realtime=False)
    ).run()

    assert report.approach_reached
    assert report.completed
    first = np.asarray(episode.commanded_joints_deg[0], float)
    assert np.abs(backend.true_joints_deg()[:5] - report.steps[0].commanded_joints_deg).max() < 10.0
    assert np.abs(report.steps[0].commanded_joints_deg - first).max() < 1e-6


def test_replay_stops_between_steps_when_asked(tmp_path, config):
    meta = record(tmp_path, config, steps=80)
    episode = EpisodeStore(tmp_path).load(meta.id)

    stop_after = 10
    seen: list = []
    replayer = EpisodeReplayer(
        episode, MockFollower(), config, ReplayConfig(mode="joint", realtime=False)
    )
    report = replayer.run(
        on_step=seen.append,
        should_continue=lambda: len(seen) < stop_after,
    )
    assert not report.completed
    assert report.n_steps == stop_after
    assert "operator" in report.aborted_reason


# ------------------------------------------------------------------ inspection


def test_inspection_is_clean_for_a_freshly_recorded_episode(tmp_path, config):
    meta = record(tmp_path, config, steps=40)
    episode = EpisodeStore(tmp_path).load(meta.id)
    assert inspect_episode(episode, config) == []


def test_inspection_reports_a_joint_order_mismatch_as_an_error(tmp_path, config):
    meta = record(tmp_path, config, steps=20)
    path = tmp_path / meta.id / META_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["joint_names"] = ["elbow_flex", "shoulder_pan", "shoulder_lift", "wrist_flex", "wrist_roll", "gripper"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    issues = inspect_episode(EpisodeStore(tmp_path).load(meta.id), config)
    assert any(issue.level == "error" and "joint order" in issue.message for issue in issues)


def test_inspection_warns_when_the_requested_speed_exceeds_the_rate_cap(tmp_path, config):
    meta = record(tmp_path, config, steps=80)
    episode = EpisodeStore(tmp_path).load(meta.id)
    peak_step = float(np.abs(np.diff(episode.commanded_joints_deg, axis=0)).max())
    # Derive the cap from the recording rather than from a magic number, so the
    # test keeps meaning "faster than this episode can safely be played" even if
    # the waveform is retuned. At 2x the recorded rate the cap must bind.
    recorded_rate = peak_step * config.teleop.control_hz
    slow = ReplayConfig(speed=2.0, max_joint_rate_deg_s=recorded_rate * 4.0)
    fast = ReplayConfig(speed=2.0, max_joint_rate_deg_s=recorded_rate)

    assert inspect_episode(episode, config, replay=slow) == []
    issues = inspect_episode(episode, config, replay=fast)
    assert any("deg/s" in issue.message for issue in issues), [i.message for i in issues]


def test_inspection_warns_before_a_simulated_episode_reaches_hardware(tmp_path, config):
    meta = record(tmp_path, config, steps=20)
    episode = EpisodeStore(tmp_path).load(meta.id)

    assert inspect_episode(episode, config, target_physical=False) == []
    issues = inspect_episode(episode, config, target_physical=True)
    assert any("simulation" in issue.message for issue in issues)


def test_replay_refuses_an_episode_with_no_frames(tmp_path, config):
    empty = Episode(
        meta=EpisodeMeta(id="ep_empty", n_steps=0),
        frames={name: np.zeros((0, 5)) for name in COLUMN_NAMES},
        path=tmp_path,
    )
    assert [i.level for i in inspect_episode(empty, config)] == ["error"]
