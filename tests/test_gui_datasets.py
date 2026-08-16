"""The dataset library and the verify/replay routes wired through the gateway.

These are the surface the new "训练集" page in the GUI speaks to. The exporter
itself is tested in `test_export.py`; this file covers the bits that were
missing on the server side until now -- listing what is on disk, re-running the
read-back check, and dispatching a dataset episode to the replayer.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from so_snake.config import SoSnakeConfig
from so_snake.data import EpisodeStore


def _write_dataset(
    root: Path,
    name: str,
    *,
    repo_id: str | None = None,
    n_episodes: int = 1,
    n_frames: int = 60,
    fps: int = 30,
    action_space: str = "delta",
    task: str = "pick",
    episode_ids: list[str] | None = None,
    verdict: dict | None = None,
) -> Path:
    """Drop a dataset-shaped directory under `root`.

    The contents are minimal -- a manifest, optionally a verdict -- enough to
    exercise the library view and the read-back. A directory without a manifest
    is also a case the GUI handles (a foreign dataset), and is exercised by
    leaving `repo_id=None`.
    """
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    if repo_id is None:
        return path
    manifest = {
        # Exactly the keys `write_manifest` produces. No `root`: a fixture that
        # invents a field is how the frontend type came to declare one that
        # nothing sent, which read as `undefined` and resolved to the dataset
        # root at the far end.
        "repo_id": repo_id,
        "task": task,
        "fps": fps,
        "action_space": action_space,
        "cameras": ["third_person", "wrist"],
        "resolution": [240, 320],
        "n_episodes": n_episodes,
        "n_frames": n_frames,
        "cancelled": False,
        "episode_ids": episode_ids or [f"src-{i}" for i in range(n_episodes)],
        "episode_root": "/tmp/episodes",
    }
    (path / "export.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if verdict is not None:
        (path / "verify.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return path


@pytest.fixture(scope="module")
def config() -> SoSnakeConfig:
    return SoSnakeConfig()


def test_datasets_lists_what_is_on_disk(tmp_path, config):
    """Each directory under the dataset root becomes one library entry.

    A directory without our `export.json` is still listed with a synthesized
    manifest built from lerobot's `meta/info.json` -- replay and partial
    verify both need the parquet + fps + action_space, and the GUI uses
    `ours` to decide whether to gate anything on the source mapping.
    """
    from so_snake.gui.exporter import Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    _write_dataset(dataset_root, "ours", repo_id="so_snake/pick", n_episodes=12, n_frames=300)
    _write_dataset(dataset_root, "theirs")  # no manifest
    (dataset_root / "not-a-dir.txt").write_text("noise")

    payload = Exporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root).datasets()

    names = sorted(entry["name"] for entry in payload["datasets"])
    assert names == ["ours", "theirs"]
    ours = next(entry for entry in payload["datasets"] if entry["name"] == "ours")
    assert ours["manifest"] is not None
    assert ours["manifest"]["n_episodes"] == 12
    assert ours["ours"] is True
    theirs = next(entry for entry in payload["datasets"] if entry["name"] == "theirs")
    assert theirs["manifest"] is None
    assert theirs["ours"] is False
    assert theirs["verdict"] is None


def test_datasets_synthesizes_manifest_from_lerobot_info_json(tmp_path, config):
    """A foreign / legacy dataset without our `export.json` still gets an entry.

    The synthesized manifest comes from `meta/info.json`: fps, action_space
    (inferred from feature names), and an empty `episode_ids`. The GUI uses
    `ours=false` to know the source mapping is unknown.
    """
    import json

    from so_snake.gui.exporter import Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    path = dataset_root / "legacy"
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

    [entry] = Exporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root).datasets()["datasets"]
    assert entry["manifest"]["fps"] == 26
    assert entry["manifest"]["action_space"] == "delta"
    assert entry["manifest"]["episode_ids"] == []
    assert entry["ours"] is False


def test_datasets_infer_action_space_from_feature_names(tmp_path, config):
    """`dx/dy/...` -> delta, plain `x/y/...` -> absolute.

    What the exporter writes, and what `apply_action` reads; recording it on
    the dataset lets the synthesis stay in agreement with the rest of the
    pipeline.
    """
    import json

    from so_snake.gui.exporter import Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()

    def add(name: str, action_names: list[str]) -> None:
        path = dataset_root / name
        (path / "meta").mkdir(parents=True)
        (path / "meta" / "info.json").write_text(
            json.dumps(
                {
                    "fps": 30,
                    "features": {
                        "action": {"names": action_names},
                        "observation.state": {"names": ["x", "y", "z", "pitch", "roll", "gripper"]},
                    },
                }
            ),
            encoding="utf-8",
        )

    add("delta_ds", ["dx", "dy", "dz", "dpitch", "droll", "gripper"])
    add("absolute_ds", ["x", "y", "z", "pitch", "roll", "gripper"])

    payload = Exporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root).datasets()
    by_name = {e["name"]: e for e in payload["datasets"]}
    assert by_name["delta_ds"]["manifest"]["action_space"] == "delta"
    assert by_name["absolute_ds"]["manifest"]["action_space"] == "absolute"


def test_dataset_verdict_is_marked_stale_when_files_change(tmp_path, config):
    """A dataset re-exported in place must not stay green over changed bytes."""
    from so_snake.gui.exporter import Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    path = _write_dataset(dataset_root, "ours", repo_id="so_snake/pick")
    from so_snake.gui.exporter import VERDICT_VERSION

    (path / "verify.json").write_text(
        json.dumps(
            {
                # This version, so what is under test is staleness and not the
                # schema check -- they are different reasons to distrust a
                # cached verdict and each has its own test.
                "version": VERDICT_VERSION,
                "verified_at": time.time() - 100,
                "verified_mtime": 0.0,  # older than the dataset -> stale
                "ok": True,
                "issues": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    payload = Exporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root).datasets()
    [entry] = payload["datasets"]
    assert entry["verdict"] is not None
    assert entry["verdict"]["ok"] is True
    assert entry["verdict"]["stale"] is True


def test_dataset_replay_dispatches_to_the_session(tmp_path, config):
    """`start_dataset_replay` is the bridge between the dataset page and the arm.

    The actual arm driving lives in `SessionManager` and is tested separately.
    Here we only need to know that a request body is shaped into a call, and
    that the rig (backend, port, joint map) rides along.
    """
    from so_snake.gui.server import Gateway

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    path = _write_dataset(dataset_root, "ours", repo_id="so_snake/pick")
    gateway = Gateway(config, episode_root=tmp_path / "episodes", dataset_root=dataset_root)
    captured: dict = {}

    def fake_start_dataset_replay(target_path, index, spec, replay):
        captured["path"] = target_path
        captured["index"] = index
        captured["spec"] = spec
        captured["replay"] = replay
        return gateway.session.status()

    gateway.session.start_dataset_replay = fake_start_dataset_replay  # type: ignore[assignment]

    body = {
        "dataset": str(path),
        "episode_index": 2,
        "backend": "mock",
        "speed": 0.5,
    }
    gateway.start_dataset_replay(body)

    assert captured["path"] == path
    assert captured["index"] == 2
    assert captured["spec"].backend == "mock"
    # mode is forced to task: a dataset has no joint stream, and joint mode
    # would replay the IK's own arithmetic.
    assert captured["replay"].mode == "task"
    assert captured["replay"].speed == pytest.approx(0.5)


def test_dataset_replay_rejects_path_traversal(tmp_path, config):
    """`resolve` is the only line of defence against `../` escaping the root."""
    from so_snake.gui.exporter import Exporter
    from so_snake.gui.server import Gateway

    gateway = Gateway(config, episode_root=tmp_path)
    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    (tmp_path / "secret.txt").write_text("nope")

    with pytest.raises(ValueError, match="not inside"):
        Exporter(gateway.session.store, config, dataset_root=dataset_root).resolve("../secret.txt")


def test_dataset_replay_refuses_while_the_arm_is_driven(tmp_path, config, monkeypatch):
    """Same rule as a take replay: one thing drives the arm at a time."""
    from so_snake.gui.server import Gateway

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    path = _write_dataset(dataset_root, "ours", repo_id="so_snake/pick")
    gateway = Gateway(config, episode_root=tmp_path / "episodes", dataset_root=dataset_root)
    monkeypatch.setattr(type(gateway.session), "busy", property(lambda self: True))
    monkeypatch.setattr(type(gateway.session), "is_held", property(lambda self: False))
    monkeypatch.setattr(type(gateway.session), "mode", property(lambda self: "teleop"))

    with pytest.raises(RuntimeError, match="busy"):
        gateway.start_dataset_replay({"dataset": str(path), "backend": "mock"})


def test_start_verify_runs_the_job_and_writes_a_verdict(tmp_path, config):
    """`start_verify` is the GUI's "re-run the read-back check" button.

    The job runs on its own thread and writes a `verify.json` next to the
    dataset on success. We don't actually run `verify` here (it walks parquet
    and video; the export tests cover that path); instead we patch the
    exporter's runner to record that it was called and to write a verdict
    file, and assert the surface the GUI uses.
    """
    from so_snake.gui.exporter import Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    path = _write_dataset(dataset_root, "ours", repo_id="so_snake/pick")

    class StubExporter(Exporter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls: list[Path] = []

        def _run_verify(self, target: Path) -> None:  # type: ignore[override]
            self.calls.append(target)
            (target / "verify.json").write_text(
                json.dumps(
                    {
                        "verified_at": time.time(),
                        "verified_mtime": target.stat().st_mtime,
                        "ok": True,
                        "issues": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            with self._lock:
                self._progress.verify_report = {"ok": True, "issues": []}
                self._progress.phase = "done"

    exporter = StubExporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root)
    progress = exporter.start_verify(str(path))

    assert progress["phase"] == "verifying"
    # The job is on its own thread: wait for it, but bound the wait so a
    # regression that hangs does not hang the test.
    thread = exporter._thread
    assert thread is not None
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert exporter.calls == [path]
    assert (path / "verify.json").is_file()


def test_start_verify_refuses_while_another_job_is_running(tmp_path, config):
    """Same slot as the export: both decode every video; running them
    together would only make each slower."""
    from so_snake.gui.exporter import Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    path = _write_dataset(dataset_root, "ours", repo_id="so_snake/pick")
    other = _write_dataset(dataset_root, "other", repo_id="so_snake/place")

    exporter = Exporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root)
    started = threading.Event()
    release = threading.Event()
    original_run = exporter._run_verify

    def blocking_run(target):
        started.set()
        # Wait for the test to release us, but exit cleanly on a hang so the
        # process is not stuck if the test's release path regresses.
        release.wait(timeout=5.0)
        return original_run(target)

    exporter._run_verify = blocking_run  # type: ignore[assignment]
    exporter.start_verify(str(path))
    assert started.wait(timeout=2.0)

    with pytest.raises(RuntimeError, match="already running"):
        exporter.start_verify(str(other))

    release.set()
    exporter._thread.join(timeout=5.0)


def test_a_verdict_from_a_different_schema_is_discarded(tmp_path, config):
    """A cached verdict is an answer computed by a particular version of `verify`.

    When that code changes the old answer is not merely missing a field -- it is
    a claim made by checks that no longer exist in the form they ran. This
    already bit once: `skipped` was added to the report, and verdicts written
    before it lacked the field that separates OK from PARTIAL, so an old file
    would have shown a clean green badge for a check that never ran.
    """
    from so_snake.gui.exporter import VERDICT_VERSION, Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    path = _write_dataset(dataset_root, "ours", repo_id="so_snake/pick")
    (path / "verify.json").write_text(
        json.dumps({"version": VERDICT_VERSION + 1, "ok": True, "issues": []}),
        encoding="utf-8",
    )

    exporter = Exporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root)
    [entry] = exporter.datasets()["datasets"]
    # Not verified by the code now running, which is the same thing to the
    # operator as never verified -- and honest, unlike a green badge.
    assert entry["verdict"] is None


def test_the_dataset_root_is_not_itself_a_dataset(tmp_path, config):
    """The bug behind "meta/info.json is missing" pointing at the root.

    `Path("")` and `Path(".")` both join to the root, so a request that omitted
    the dataset name -- which is what an undefined field serialises to, since
    JSON drops the key entirely -- used to resolve to the container and have it
    opened as a dataset. The error then surfaced three layers down as lerobot
    metadata missing from `data/lerobot`, which is true and explains nothing.
    """
    from so_snake.gui.exporter import Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    _write_dataset(dataset_root, "ours", repo_id="so_snake/pick")
    exporter = Exporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root)

    for missing in ("", "   ", ".", "/"):
        with pytest.raises(ValueError, match="no dataset was named"):
            exporter.resolve(missing)

    with pytest.raises(ValueError, match="is the dataset root"):
        exporter.resolve(str(dataset_root))

    # And the normal case still works, by name and by absolute path.
    assert exporter.resolve("ours") == (dataset_root / "ours").resolve()
    assert exporter.resolve(str(dataset_root / "ours")) == (dataset_root / "ours").resolve()


def test_a_foreign_dataset_reports_its_episode_count(tmp_path, config):
    """The episode picker reads `n_episodes`; without it replay is unreachable.

    A dataset with no `export.json` gets a manifest synthesised from lerobot's
    `info.json`. That synthesised dict must have the *same keys* as the real
    one, or the UI reads a field that exists on only one of them -- which is
    what happened: `n_episodes` was absent, the picker showed zero episodes,
    and a foreign dataset could not be replayed at all.
    """
    from so_snake.data.export import dataset_meta
    from so_snake.gui.exporter import Exporter

    dataset_root = tmp_path / "lerobot"
    dataset_root.mkdir()
    foreign = dataset_root / "foreign"
    (foreign / "meta").mkdir(parents=True)
    (foreign / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 26,
                "total_episodes": 10,
                "total_frames": 4221,
                "features": {
                    "action": {"names": ["dx", "dy", "dz", "dpitch", "droll", "gripper"]},
                    "observation.images.wrist": {"shape": [240, 320, 3]},
                    "observation.images.third_person": {"shape": [240, 320, 3]},
                },
            }
        ),
        encoding="utf-8",
    )

    synthesised, ours = dataset_meta(foreign)
    assert not ours
    assert synthesised["n_episodes"] > 0, "the picker would show zero episodes"

    ours_path = _write_dataset(dataset_root, "ours", repo_id="so_snake/pick")
    real, is_ours = dataset_meta(ours_path)
    assert is_ours
    # One shape, so nothing downstream has to ask which kind it got.
    assert sorted(synthesised) == sorted(real)
    assert "root" not in real, "where a dataset lives is not part of its metadata"

    exporter = Exporter(EpisodeStore(tmp_path / "episodes"), config, dataset_root=dataset_root)
    by_name = {d["name"]: d for d in exporter.datasets()["datasets"]}
    assert by_name["foreign"]["manifest"]["n_episodes"] > 0
    assert by_name["foreign"]["ours"] is False
