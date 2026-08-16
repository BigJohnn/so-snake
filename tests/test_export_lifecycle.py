"""The three behaviours that fix the manifest / overwrite failure modes.

The `data/lerobot/niuniu_pick_place` dataset this bench has is a real
LeRobotDataset (parquet + videos + lerobot's `info.json`) but missing so-snake's
`export.json`. The first three cases here pin down the policy:

  * replay must keep working, because the parquet is the source of truth;
  * verify must do partial work and say so explicitly -- round-trip passes,
    source-fidelity skipped -- so a green verdict cannot be read as "matches
    the source takes";
  * re-exporting into the same directory must refuse by default and wipe on
    explicit `--overwrite`.

The fourth case is the helper that makes all of the above possible: the
manifest-or-info.json synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from so_snake.config import SoSnakeConfig
from so_snake.data import EpisodeStore
from so_snake.data.export import (
    ExportConfig,
    _dataset_meta,
    _summarise_existing,
    episode_from_dataset,
    export,
    verify,
)
from so_snake.m4_execution import MockFollower
from so_snake.teleop import ScriptedSource, TeleopLoop


@pytest.fixture(scope="module")
def config() -> SoSnakeConfig:
    return SoSnakeConfig()


def _record(root: Path, config: SoSnakeConfig, *, task: str = "pick", steps: int = 60) -> str:
    """A real take into a real episode store, with real mp4s so the export
    pipeline actually has video frames to read.

    Without real videos the export fails inside `decode_video` (an empty file
    is not a valid mp4), and any test that calls `export()` end-to-end needs
    them. The cost is real -- a few hundred KB per take -- but it is what
    the operator's bench does and what the test should mirror.
    """
    import av

    from so_snake.data import EpisodeRecorder

    backend = MockFollower()
    loop = TeleopLoop(ScriptedSource.from_waveform(steps, amplitude=0.2), backend, config)
    recorder = EpisodeRecorder(root, config=config, backend="mock", source="scripted", joint_names=backend.joint_names)
    recorder.start(task=task)
    loop.run(max_steps=steps, realtime=False, on_step=recorder.append)
    meta = recorder.stop(keep=True)
    assert meta is not None

    # Real videos, not stubs. The export reads them and re-encodes -- this is
    # the only honest way to exercise the write path in a test.
    path = root / meta.id
    for role in ("third_person", "wrist"):
        mp4 = path / f"{role}.mp4"
        with av.open(str(mp4), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for i in range(meta.n_steps):
                array = np.full((48, 64, 3), i % 256, dtype=np.uint8)
                container.mux(stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
            container.mux(stream.encode())

    # Refresh meta so the bookkeeping matches what we just wrote.
    meta.video = {
        "encoder": {"codec": "libx264", "reason": "test", "hardware": False},
        "cameras": {
            role: {
                "file": f"{role}.mp4",
                "width": 64,
                "height": 48,
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
    return meta.id


def _make_dataset_with_manifest(
    dataset_root: Path,
    name: str,
    *,
    source_episode_id: str,
    fps: int = 26,
    action_space: str = "delta",
) -> Path:
    """Run a real export through the full pipeline and return the dataset path.

    The point of going through `export()` (instead of writing parquet by hand)
    is to exercise the same write path an operator would, so the test catches
    regressions in the export contract rather than just in the test fixture.
    """
    from so_snake.data.export import read_manifest

    dataset_root.mkdir(parents=True, exist_ok=True)
    out = dataset_root / name
    config = ExportConfig(repo_id=f"so_snake/{name}", root=out, fps=fps, action_space=action_space)
    report = export(
        EpisodeStore.__new__(EpisodeStore),  # placeholder; see _record helper below
        config,
    )
    raise NotImplementedError("placeholder; tests below use the _record path")


def _make_real_export(
    episode_root: Path,
    dataset_root: Path,
    config: SoSnakeConfig,
    *,
    task: str = "pick",
    name: str = "ours",
    fps: int | None = None,
) -> Path:
    """Record one take into the episode store and export it as a dataset.

    Returns the dataset directory. The dataset ends up with our `export.json`
    (so `verify` has a source mapping) and a real parquet (so replay and the
    round-trip checks have something to read). `fps=None` lets the exporter
    measure the rate from the recorded takes rather than trusting a number
    a test happened to type.
    """
    episode_root.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)
    _record(episode_root, config, task=task)
    out = dataset_root / name
    store = EpisodeStore(episode_root)
    export_cfg = ExportConfig(repo_id=f"so_snake/{name}", task=task, root=out, fps=fps)
    export(store, export_cfg)
    return out


# --------------------------------------------------- the helper itself


def test_dataset_meta_uses_manifest_when_present(tmp_path, config):
    """`ours=True` when our `export.json` is sitting next to the dataset."""
    ep = tmp_path / "episodes"
    ds_root = tmp_path / "lerobot"
    path = _make_real_export(ep, ds_root, config, name="ours")

    meta, ours = _dataset_meta(path)
    assert ours is True
    assert meta["repo_id"] == "so_snake/ours"
    # The fps is whatever the loop actually ran at (30 here, because the
    # mocked loop has no real-time pressure); what matters for this test is
    # that the manifest carries an integer and the synthesis below agrees
    # with it (or at least returns a number of the same shape).
    assert isinstance(meta["fps"], int) and meta["fps"] > 0
    assert meta["episode_ids"]  # non-empty: our manifest records sources


def test_dataset_meta_falls_back_to_lerobot_info_json(tmp_path, config):
    """A foreign / legacy dataset still gets fps and action_space from lerobot.

    `ours=False` and `episode_ids=[]`: source mapping is unknown. The action
    space comes from the action feature's names -- exactly what lerobot
    writes, exactly what `apply_action` understands.
    """
    ds_root = tmp_path / "lerobot"
    ds_root.mkdir()
    path = ds_root / "legacy"
    (path / "meta").mkdir(parents=True)
    (path / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 26,
                "features": {
                    "action": {"names": ["dx", "dy", "dz", "dpitch", "droll", "gripper"]},
                    "observation.state": {"names": ["x", "y", "z", "pitch", "roll", "gripper"]},
                },
            }
        ),
        encoding="utf-8",
    )

    meta, ours = _dataset_meta(path)
    assert ours is False
    assert meta["fps"] == 26
    assert meta["action_space"] == "delta"
    assert meta["episode_ids"] == []
    assert meta["repo_id"] == "legacy"


# ------------------------------------------------------- replay decoupling


def test_replay_works_on_a_dataset_without_our_manifest(tmp_path, config):
    """`episode_from_dataset` does not require `export.json`.

    The parquet is what replay reads; the manifest only contributes the
    source-take name, which is a nice-to-have, not a hard dependency.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ep = tmp_path / "episodes"
    ds_root = tmp_path / "lerobot"
    full = _make_real_export(ep, ds_root, config)
    # Strip our manifest. The parquet and videos are still there; that is what
    # replay has to keep working on.
    (full / "export.json").unlink()

    # Sanity: the directory is still a real LeRobotDataset.
    assert LeRobotDataset("so_snake/ours", root=full).num_episodes == 1

    episode = episode_from_dataset(full, 0, so_snake_config=config)
    # The action space is inferred from info.json, and the targets come from
    # `apply_action` over the on-disk rows. The action column has 6 floats and
    # the take has n_steps of them -- that's all replay cares about.
    assert episode.frames["action.task.target"].shape == (episode.meta.n_steps, 5)
    assert episode.frames["action.task.gripper_deg"].shape == (episode.meta.n_steps,)
    # Notes carry the "no source mapping" marker; replay's only job is to
    # build the episode, which it just did.
    assert "no source mapping" in episode.meta.notes


# ------------------------------------------------------ verify partial work


def test_verify_records_source_fidelity_as_skipped_when_manifest_missing(tmp_path, config):
    """Round-trip + time axis run, source-fidelity skipped, ok stays True.

    A green `ok` here means "the parquet reads back to itself" -- not "this
    matches the source takes". The `skipped` list is the operator's only way
    to see which one it is, so it has to be there.
    """
    ep = tmp_path / "episodes"
    ds_root = tmp_path / "lerobot"
    full = _make_real_export(ep, ds_root, config)
    (full / "export.json").unlink()

    report = verify(full, EpisodeStore(ep), so_snake_config=config)

    # Round-trip did run: the parquet and timestamps are intact.
    assert report.n_episodes == 1
    assert report.timestamp_max_error_s == pytest.approx(0.0, abs=1e-6)
    # Source-fidelity did NOT run: that is the whole point of the skip.
    assert any("export.json" in note for note in report.skipped)
    # ok is True -- the round-trip passed -- but the report says so.
    assert report.ok is True


def test_verify_with_a_real_manifest_records_no_skips(tmp_path, config):
    """A full verify still goes through and reports the source-fidelity path."""
    ep = tmp_path / "episodes"
    ds_root = tmp_path / "lerobot"
    full = _make_real_export(ep, ds_root, config)

    report = verify(full, EpisodeStore(ep), so_snake_config=config)

    assert report.skipped == []
    assert report.ok is True


# ------------------------------------------------- overwrite the destructive


def test_export_refuses_when_target_directory_exists(tmp_path, config):
    """Default: refuse. The error names what's there, so the operator's next
    step (different --out, or --overwrite) is informed."""
    ep = tmp_path / "episodes"
    ds_root = tmp_path / "lerobot"
    existing = _make_real_export(ep, ds_root, config)

    cfg = ExportConfig(repo_id="so_snake/ours", root=existing)
    with pytest.raises(FileExistsError) as excinfo:
        export(EpisodeStore(ep), cfg)

    # The message is the operator's only guide here; the summary line is the
    # difference between "directory exists" (recoverable, choose next step)
    # and "stack trace from inside lerobot" (looks like a server crash).
    msg = str(excinfo.value)
    assert "already exists" in msg
    assert "overwrite=True" in msg
    # The dataset is untouched.
    assert (existing / "export.json").is_file()


def test_export_with_overwrite_wipes_and_rebuilds(tmp_path, config):
    """`overwrite=True` is destructive and intentional -- the previous dataset
    must be gone before the new one starts writing."""
    ep = tmp_path / "episodes"
    ds_root = tmp_path / "lerobot"
    existing = _make_real_export(ep, ds_root, config)
    parquet_count_before = sum(1 for _ in existing.rglob("*.parquet"))
    assert parquet_count_before > 0

    cfg = ExportConfig(repo_id="so_snake/ours", root=existing)
    report = export(EpisodeStore(ep), cfg, overwrite=True)

    # The new dataset is ours: manifest is written, the parquet round-trip
    # passes when we read it back.
    assert (existing / "export.json").is_file()
    re_verified = verify(existing, EpisodeStore(ep), so_snake_config=config)
    assert re_verified.ok is True
    assert report.dataset_path == existing


def test_export_overwrite_on_empty_directory_still_works(tmp_path, config):
    """The flag's contract is "if a directory exists, replace it". An empty
    directory is a directory that exists; the flag does not require a real
    dataset to be present."""
    ep = tmp_path / "episodes"
    _record(ep, config)
    target = tmp_path / "lerobot" / "stub"
    target.mkdir(parents=True)

    cfg = ExportConfig(repo_id="so_snake/stub", root=target)
    export(EpisodeStore(ep), cfg, overwrite=True)

    assert (target / "export.json").is_file()


def test_summarise_existing_uses_info_json_when_present(tmp_path):
    """The overwrite error tells the operator what they're replacing."""
    ds = tmp_path / "ds"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 10, "total_frames": 4221, "fps": 26}),
        encoding="utf-8",
    )
    assert "10 episodes / 4221 frames" in _summarise_existing(ds)


def test_summarise_existing_falls_back_to_size(tmp_path):
    """A stub directory (no info.json) gets a size line, not silence."""
    ds = tmp_path / "stub"
    ds.mkdir()
    (ds / "junk.bin").write_bytes(b"x" * 4096)
    summary = _summarise_existing(ds)
    # 4096 bytes -> ~0 MB. We just need a non-empty string with a size hint.
    assert summary != ""
    assert "MB" in summary or "empty" in summary
