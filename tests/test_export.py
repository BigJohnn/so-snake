"""The LeRobotDataset export: the action-space contract and the screening.

Nothing here needs lerobot. `plan` and `build_state_action` are the whole of the
export's decision-making, and they run on numpy alone; the dataset write is a
thin loop over what they return. Keeping the tests on that side of the lazy
import is what lets the offline gate cover the export at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from so_snake.config import SoSnakeConfig
from so_snake.data import (
    EpisodeRecorder,
    EpisodeStore,
    ExportConfig,
    apply_action,
    build_state_action,
    measured_fps,
    observed_task_pose,
    plan,
)
from so_snake.data.export import STATE_DIM, resolve_fps, select_episodes
from so_snake.m4_execution import MockFollower
from so_snake.teleop import ScriptedSource, TeleopLoop


@pytest.fixture(scope="module")
def config() -> SoSnakeConfig:
    return SoSnakeConfig()


def record(root, config, *, steps: int = 60, task: str = "pick", cameras: bool = True):
    """A real loop into a real episode, with fake camera bookkeeping if asked.

    The videos themselves are not written -- the screening reads the counts out
    of `meta.video`, and the decode is exercised separately.
    """
    backend = MockFollower()
    loop = TeleopLoop(ScriptedSource.from_waveform(steps, amplitude=0.2), backend, config)
    recorder = EpisodeRecorder(
        root, config=config, backend="mock", source="scripted", joint_names=backend.joint_names
    )
    recorder.start(task=task)
    loop.run(max_steps=steps, realtime=False, on_step=recorder.append)
    meta = recorder.stop(keep=True)

    if cameras:
        import json

        path = root / meta.id
        meta.video = {
            "encoder": {"codec": "libsvtav1", "reason": "test", "hardware": False},
            "cameras": {
                role: {
                    "file": f"{role}.mp4",
                    "width": 640,
                    "height": 480,
                    "written": meta.n_steps,
                    "dropped": 0,
                    "stale": 0,
                    "error": "",
                }
                for role in ("third_person", "wrist")
            },
        }
        (path / "meta.json").write_text(
            json.dumps(meta.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for role in ("third_person", "wrist"):
            (path / f"{role}.mp4").write_bytes(b"")
    return meta


def at_rate(root, meta, hz: float) -> None:
    """Rewrite a recorded take so it reads as having run at `hz`.

    Both the step periods and the wall-clock duration, because a real take has
    both and they agree. The rate is read from the *periods* -- see
    `Episode.measured_hz`, which moved off `n_steps / duration_s` because one
    startup stall in an otherwise clean take skewed that average by 6% -- so a
    fixture that only rewrote the duration would be lying about the wrong thing.
    """
    import json

    from so_snake.data.episode import FRAMES_NAME, META_NAME

    path = root / meta.id
    with np.load(path / FRAMES_NAME) as data:
        frames = {name: data[name] for name in data.files}
    frames["diagnostics.loop_dt_s"] = np.full(len(frames["t"]), 1.0 / hz)
    np.savez_compressed(path / FRAMES_NAME, **frames)

    payload = json.loads((path / META_NAME).read_text(encoding="utf-8"))
    payload["duration_s"] = payload["n_steps"] / hz
    (path / META_NAME).write_text(json.dumps(payload), encoding="utf-8")


# ----------------------------------------------------------- the column fix


def test_measured_gripper_is_recorded(tmp_path, config):
    """v1 threw the bus reading away; v2 keeps it, distinct from the command."""
    meta = record(tmp_path, config, cameras=False)
    episode = EpisodeStore(tmp_path).load(meta.id)

    gripper, was_measured = episode.measured_gripper_deg()
    assert was_measured
    assert gripper.shape == (meta.n_steps,)
    assert episode.meta.format_version >= 2


def test_v1_episode_falls_back_and_says_so(tmp_path, config):
    """An episode without the column stays readable, and admits the substitution."""
    meta = record(tmp_path, config, cameras=False)
    episode = EpisodeStore(tmp_path).load(meta.id)
    del episode.frames["observation.state.gripper_deg"]

    gripper, was_measured = episode.measured_gripper_deg()
    assert not was_measured
    np.testing.assert_allclose(gripper, episode.gripper_cmd_deg)


# ------------------------------------------------------- state and action


def test_state_is_the_reached_pose_not_the_commanded_one(tmp_path, config):
    """The exported state must come from the measurement, not the IK solution."""
    meta = record(tmp_path, config, cameras=False)
    episode = EpisodeStore(tmp_path).load(meta.id)

    state, _, _ = build_state_action(episode, "delta", config)
    assert state.shape == (meta.n_steps, STATE_DIM)
    np.testing.assert_allclose(
        state[:, :5], observed_task_pose(episode, config), rtol=0, atol=1e-6
    )


def test_delta_action_round_trips_through_apply_action(tmp_path, config):
    """Training and rollout must agree on the anchor, or the arm creeps.

    `apply_action` is what the rollout runner will call; feeding it the state
    the exporter paired with each action has to reproduce the recorded target.
    """
    meta = record(tmp_path, config, cameras=False)
    episode = EpisodeStore(tmp_path).load(meta.id)
    state, action, _ = build_state_action(episode, "delta", config)
    target = episode.task_target

    # Row i's action is anchored on row i-1's pose, which is the pose a rollout
    # holds when it is about to ask for row i.
    for i in range(1, meta.n_steps):
        recovered, gripper = apply_action(state[i - 1, :5], action[i], "delta")
        np.testing.assert_allclose(recovered, target[i], rtol=0, atol=1e-5)
        assert gripper == pytest.approx(episode.gripper_cmd_deg[i])


def test_absolute_action_round_trips_too(tmp_path, config):
    meta = record(tmp_path, config, cameras=False)
    episode = EpisodeStore(tmp_path).load(meta.id)
    state, action, _ = build_state_action(episode, "absolute", config)

    for i in range(meta.n_steps):
        recovered, _ = apply_action(state[i, :5], action[i], "absolute")
        np.testing.assert_allclose(recovered, episode.task_target[i], rtol=0, atol=1e-5)


def test_angular_action_components_are_wrapped(tmp_path, config):
    """A roll crossing +-180 must not export a 2*pi step."""
    meta = record(tmp_path, config, cameras=False)
    episode = EpisodeStore(tmp_path).load(meta.id)

    # Put the target just past +pi and the pose just short of it: the naive
    # difference is ~2*pi, the manifold step is ~0.
    episode.frames["action.task.target"][:, 4] = -np.pi + 0.01
    episode.frames["observation.state.joints_deg"][:, 4] = 0.0

    _, action, _ = build_state_action(episode, "delta", config)
    assert np.abs(action[:, 4]).max() <= np.pi


def test_gripper_stays_absolute_in_the_delta_space(tmp_path, config):
    meta = record(tmp_path, config, cameras=False)
    episode = EpisodeStore(tmp_path).load(meta.id)

    _, action, _ = build_state_action(episode, "delta", config)
    np.testing.assert_allclose(action[:, 5], episode.gripper_cmd_deg, rtol=0, atol=1e-4)


# ------------------------------------------------------------ frame rate


def test_fps_is_measured_from_the_clock_not_the_config(tmp_path, config):
    """The loop is configured for 30 Hz; the export follows what it held.

    A rollout at the configured rate runs faster than every demonstration behind
    it, on an arm whose tracking lag is already the largest term in the action.
    """
    meta = record(tmp_path, config, cameras=True)
    at_rate(tmp_path, meta, 26.0)
    episode = EpisodeStore(tmp_path).load(meta.id)

    assert measured_fps(episode) == pytest.approx(26.0)
    assert episode.meta.control_hz == 30.0, "the config still says 30"

    export_config = ExportConfig(repo_id="t/t", task="pick")
    assert resolve_fps([episode], export_config) == 26
    assert export_config.fps is None


def test_one_stalled_step_does_not_move_the_dataset_rate(tmp_path, config):
    """The reason the rate is the median period and not `n_steps / duration_s`.

    Choosing an encoder costs ~700 ms and used to happen on the first frame of
    a take, under the lock the control loop needs. On a real 292-step take that
    single step pulled the average from 30.1 Hz to 28.2 -- so the exported
    dataset would have laid all 292 frames on a 28 Hz grid because of one of
    them. The median reports the period the other 291 actually ran at.
    """
    meta = record(tmp_path, config, steps=100, cameras=True)
    at_rate(tmp_path, meta, 30.0)

    from so_snake.data.episode import FRAMES_NAME

    path = tmp_path / meta.id
    with np.load(path / FRAMES_NAME) as data:
        frames = {name: data[name] for name in data.files}
    frames["diagnostics.loop_dt_s"][1] = 0.711  # the encoder probe, measured
    np.savez_compressed(path / FRAMES_NAME, **frames)

    episode = EpisodeStore(tmp_path).load(meta.id)
    # What the old metric would have said, for contrast.
    stalled_average = episode.meta.n_steps / (
        float(frames["diagnostics.loop_dt_s"][1:].sum())
    )
    assert stalled_average < 29.0, "the stall really does drag the average"

    assert measured_fps(episode) == pytest.approx(30.0, rel=1e-6)
    assert resolve_fps([episode], ExportConfig(repo_id="t/t")) == 30


def test_explicit_fps_is_honoured(tmp_path, config):
    meta = record(tmp_path, config, cameras=True)
    episode = EpisodeStore(tmp_path).load(meta.id)
    assert resolve_fps([episode], ExportConfig(repo_id="t/t", fps=10)) == 10


# ------------------------------------------------------------- screening


def test_task_selection_takes_only_that_task(tmp_path, config):
    record(tmp_path, config, task="pick", cameras=True)
    record(tmp_path, config, task="place", cameras=True)
    store = EpisodeStore(tmp_path)

    chosen = select_episodes(store, ExportConfig(repo_id="t/t", task="pick"))
    assert [e.meta.task for e in chosen] == ["pick"]
    assert len(select_episodes(store, ExportConfig(repo_id="t/t"))) == 2


def test_misaligned_video_is_rejected_not_truncated(tmp_path, config):
    """A video short of the frame table has lost row alignment. Refuse it."""
    import json

    meta = record(tmp_path, config, cameras=True)
    path = tmp_path / meta.id
    payload = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    payload["video"]["cameras"]["wrist"]["written"] = meta.n_steps - 3
    (path / "meta.json").write_text(json.dumps(payload), encoding="utf-8")

    report, usable = plan(EpisodeStore(tmp_path), ExportConfig(repo_id="t/t", task="pick"))
    assert not usable
    assert "wrist wrote" in report.skipped[0].reason


def test_episode_without_a_requested_camera_is_rejected(tmp_path, config):
    record(tmp_path, config, task="pick", cameras=False)
    report, usable = plan(EpisodeStore(tmp_path), ExportConfig(repo_id="t/t", task="pick"))
    assert not usable
    assert "no third_person camera" in report.skipped[0].reason


def test_off_rate_episode_is_rejected_from_a_shared_timeline(tmp_path, config):
    """One integer rate covers the dataset, so a take at a different speed is
    out -- exporting it anyway would stretch its motion onto the wrong grid."""
    good = [record(tmp_path, config, task="pick", cameras=True) for _ in range(3)]
    slow = record(tmp_path, config, task="pick", cameras=True)
    for meta in good:
        at_rate(tmp_path, meta, 26.0)
    at_rate(tmp_path, slow, 12.0)

    report, usable = plan(EpisodeStore(tmp_path), ExportConfig(repo_id="t/t", task="pick"))
    assert report.fps == 26  # the median holds, the outlier does not move it
    assert sorted(e.meta.id for e, _, _ in usable) == sorted(m.id for m in good)
    assert "Hz, dataset is" in report.skipped[0].reason


def test_plan_reports_zero_action_share(tmp_path, config):
    record(tmp_path, config, task="pick", cameras=True)
    report, _ = plan(EpisodeStore(tmp_path), ExportConfig(repo_id="t/t", task="pick"))
    assert 0.0 <= report.action_stats["all_zero_frac"] <= 1.0
    assert set(report.action_stats) >= {"x", "y", "z", "pitch", "roll", "gripper"}


def test_gripper_is_reported_as_a_duty_cycle_not_a_step_size(tmp_path, config):
    """Percentiles of an absolute angle describe how long the jaw was open and
    cannot say whether the takes contain a grasp; crossings can."""
    meta = record(tmp_path, config, task="pick", cameras=True)
    store = EpisodeStore(tmp_path)

    report, _ = plan(store, ExportConfig(repo_id="t/t", task="pick"))
    grip = report.action_stats["gripper"]
    assert set(grip) == {"min", "max", "closed_frac", "transitions"}

    # A take that never closes has to be visibly graspless.
    episode = store.load(meta.id)
    episode.frames["action.task.gripper_deg"][:] = 90.0
    _, action, _ = build_state_action(episode, "delta", config)
    from so_snake.data.export import _action_stats

    assert _action_stats(action)["gripper"]["transitions"] == 0


def test_no_match_is_an_error_not_an_empty_dataset(tmp_path, config):
    record(tmp_path, config, task="pick", cameras=True)
    with pytest.raises(ValueError, match="no episodes matched"):
        plan(EpisodeStore(tmp_path), ExportConfig(repo_id="t/t", task="nonexistent"))


def test_unknown_action_space_is_refused():
    with pytest.raises(ValueError, match="action_space"):
        ExportConfig(repo_id="t/t", action_space="joint")


# --------------------------------------------------------------- decoding

av = pytest.importorskip("av", reason="video decode needs PyAV, which ships with .[sim]")


def write_video(path, n_frames: int, size=(64, 48)) -> None:
    width, height = size
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=26)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(n_frames):
            array = np.full((height, width, 3), i % 256, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        container.mux(stream.encode())


def test_decode_resizes_and_yields_exactly_what_was_asked(tmp_path):
    from so_snake.data.export import decode_video

    path = tmp_path / "wrist.mp4"
    write_video(path, 20)

    frames = list(decode_video(path, 12, (240, 320)))
    assert len(frames) == 12
    assert frames[0].shape == (240, 320, 3)
    assert frames[0].dtype == np.uint8


def test_short_video_is_caught_by_the_frame_the_caller_asks_for(tmp_path):
    """The guard has to fire through the caller's access pattern, not only on a
    full iteration: `export` calls `next` exactly n_steps times, so a video that
    ends early must raise on that call rather than quietly stopping short."""
    from so_snake.data.export import decode_video

    path = tmp_path / "wrist.mp4"
    write_video(path, 8)

    stream = decode_video(path, 10, (48, 64))
    with pytest.raises(ValueError, match="no longer aligned"):
        for _ in range(10):
            next(stream)


# ------------------------------------------------------- measured vs playback


def test_playback_hz_keeps_a_credible_measurement(tmp_path, config):
    meta = record(tmp_path, config, cameras=True)
    # 26 against a configured 30: exactly the gap this bench recorded 43 takes
    # at before the pacing fix, and well inside the credible band.
    at_rate(tmp_path, meta, 26.0)
    episode = EpisodeStore(tmp_path).load(meta.id)

    assert episode.measured_hz == pytest.approx(26.0)
    assert episode.playback_hz == pytest.approx(26.0)


def test_playback_hz_refuses_an_offline_recording(tmp_path, config):
    """A take recorded with `realtime=False` has no wall-clock rate to recover.

    `record` above runs the loop offline, so this is the real thing rather than
    a contrived duration: the loop went as fast as the mock would allow, which
    is hundreds of hertz. Handing that to a replayer as the recording rate would
    ask it to play the episode back at hundreds of hertz.
    """
    meta = record(tmp_path, config, cameras=True)
    episode = EpisodeStore(tmp_path).load(meta.id)

    assert episode.measured_hz > 2 * episode.meta.control_hz
    assert episode.playback_hz == pytest.approx(episode.meta.control_hz)


# ------------------------------------------------------------- the manifest


def test_manifest_records_the_source_takes_in_dataset_order(tmp_path, config):
    """Without it the export is a one-way door; see the module docstring."""
    from so_snake.data import MANIFEST_NAME, read_manifest, write_manifest
    from so_snake.data.export import ExportReport

    dataset = tmp_path / "ds"
    dataset.mkdir()
    report = ExportReport(fps=26, action_space="delta", n_episodes=2, n_frames=100)
    report.dataset_path = dataset
    report.episode_ids = ("ep_a", "ep_b")

    path = write_manifest(report, ExportConfig(repo_id="t/t", task="pick"))
    assert path.name == MANIFEST_NAME

    manifest = read_manifest(dataset)
    assert manifest["episode_ids"] == ["ep_a", "ep_b"]
    assert manifest["fps"] == 26
    assert manifest["action_space"] == "delta"
    assert manifest["task"] == "pick"


def test_reading_a_manifest_that_is_not_there_says_what_to_do(tmp_path):
    from so_snake.data import read_manifest

    with pytest.raises(FileNotFoundError, match="Re-export"):
        read_manifest(tmp_path)


# ------------------------------------------------------- the GUI export job


def test_export_is_refused_while_the_arm_is_driven(tmp_path, config, monkeypatch):
    """Not because it would touch the arm -- it would not -- but because it
    would take the machine out from under the control loop.

    Decoding and re-encoding two video streams is the heaviest thing this
    repository does, and the loop now spins the tail of every period to hold its
    rate. An export running underneath teleop competes for cores with the thing
    that must not miss a deadline.
    """
    from so_snake.gui.server import Gateway

    gateway = Gateway(config, episode_root=tmp_path)
    monkeypatch.setattr(type(gateway.session), "busy", property(lambda self: True))
    monkeypatch.setattr(type(gateway.session), "is_held", property(lambda self: False))
    monkeypatch.setattr(type(gateway.session), "mode", property(lambda self: "teleop"))

    with pytest.raises(RuntimeError, match="busy"):
        gateway.start_export({"repo_id": "t/t", "task": "pick"})


@pytest.mark.parametrize(
    "repo_id",
    ["", "nope", "../escape/x", "a/b/c", "  /  "],
)
def test_a_repo_id_that_would_escape_the_dataset_root_is_refused(tmp_path, config, repo_id):
    """`repo_id` becomes a directory name, so it is validated, not trusted."""
    from so_snake.gui.server import Gateway

    gateway = Gateway(config, episode_root=tmp_path)
    with pytest.raises(ValueError):
        gateway.export_config_from_body({"repo_id": repo_id})


def test_export_body_clamps_the_resolution(tmp_path, config):
    """Every frame is decoded and re-encoded at this size; 8000 would fill the disk."""
    from so_snake.gui.server import Gateway

    gateway = Gateway(config, episode_root=tmp_path)
    export_config = gateway.export_config_from_body(
        {"repo_id": "t/t", "resolution": [8000, 9000]}
    )
    assert export_config.resolution == (1080, 1920)


def test_the_task_list_counts_what_each_label_would_contribute(tmp_path, config):
    """The picker is where a training set is chosen; four takes should read as four."""
    from so_snake.gui.exporter import Exporter

    record(tmp_path, config, task="pick", cameras=True)
    record(tmp_path, config, task="pick", cameras=True)
    record(tmp_path, config, task="place", cameras=True)

    tasks = {t["task"]: t for t in Exporter(EpisodeStore(tmp_path), config).tasks()["tasks"]}

    assert tasks["pick"]["takes"] == 2
    assert tasks["place"]["takes"] == 1
    assert tasks["pick"]["steps"] == 120


def test_a_take_that_missed_its_configured_rate_says_re_record(tmp_path, config):
    """The dataset rate is a property of the recording, not of the export.

    A selection of takes that ran at 26 Hz against a configured 30 exports at
    26, and no amount of re-exporting raises it. The report has to say that, or
    the next person runs the dry run again expecting a different number.
    """
    from so_snake.data.export import format_report

    meta = record(tmp_path, config, task="pick", cameras=True)
    at_rate(tmp_path, meta, 26.0)  # configured 30, held 26

    export_config = ExportConfig(repo_id="t/t", task="pick")
    report, _ = plan(EpisodeStore(tmp_path), export_config)

    entry = next(e for e in report.episodes if e.included)
    assert entry.configured_hz == 30.0
    assert entry.measured_fps == pytest.approx(26.0)

    text = format_report(report, export_config)
    assert "did not hold it" in text
    assert "Re-record" in text


def test_a_take_that_held_its_rate_says_nothing(tmp_path, config):
    """The warning must not fire on healthy takes, or it stops being read."""
    from so_snake.data.export import format_report

    meta = record(tmp_path, config, task="pick", cameras=True)
    at_rate(tmp_path, meta, 30.0)

    export_config = ExportConfig(repo_id="t/t", task="pick")
    report, _ = plan(EpisodeStore(tmp_path), export_config)

    assert "did not hold it" not in format_report(report, export_config)
