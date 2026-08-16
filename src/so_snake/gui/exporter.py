"""The dataset export, as a background job the GUI can start and watch.

Deliberately not part of `SessionManager`. That class exists to enforce one
invariant -- one thing drives the arm at a time -- and an export drives nothing:
it reads episodes off disk and writes a dataset. Folding it into `_mode` would
mean an export blocked homing, or that "busy" stopped meaning "the arm is
moving", and that word is load-bearing everywhere else in the GUI.

What it does share with the arm is the machine. Decoding two 1080p video streams
and re-encoding them is the heaviest thing this repository does, and the control
loop now spins the tail of every period to hold 30 Hz (see `so_snake.pacing`),
so an export running underneath teleop would compete with it for cores while the
operator is recording. The gateway therefore refuses to start one while the arm
is being driven -- the same rule, and for the same reason, as the camera scan.

One export at a time, cancellable between episodes, and the job outlives the
request that started it so the browser can be closed and come back.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, SoSnakeConfig
from ..data import (
    EpisodeStore,
    ExportConfig,
    ExportReport,
    VerifyReport,
    export,
    plan,
    read_manifest,
    verify,
)

# Where the GUI puts datasets when the operator does not say. Inside the repo
# next to `data/episodes`, and git-ignored: an export is a derived artefact, and
# a 20 000-frame one is not something to accidentally commit.
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "lerobot"

PHASES = ("idle", "exporting", "verifying", "done", "failed", "cancelled")

# Where a verify verdict is remembered, inside the dataset it is about.
#
# Verifying decodes every frame of every video, which on a 20 000-frame dataset
# is minutes. The answer is a property of the bytes on disk, so it does not
# change until something rewrites them -- caching it is what lets the library
# view show a verdict per dataset without re-reading all of them on every page
# load. The recorded mtime is what makes the cache honest: a dataset written to
# since it was verified shows as stale rather than as still-good.
VERDICT_NAME = "verify.json"


@dataclass
class ExportProgress:
    """What the UI polls. Small enough to send at 1 Hz without thinking."""

    phase: str = "idle"
    # "export" or "verify" -- both are long, both use this slot, and the UI
    # needs to say which one is running.
    kind: str = ""
    repo_id: str = ""
    task: str = ""
    dataset_path: str = ""
    episodes_done: int = 0
    episodes_total: int = 0
    frames_done: int = 0
    current_episode: str = ""
    error: str = ""
    # Filled in when the run finishes; the full reports, as the CLI prints them.
    report: dict[str, Any] | None = None
    verify_report: dict[str, Any] | None = None
    log: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.phase in ("exporting", "verifying")


class Exporter:
    """Runs one export at a time, on its own thread."""

    def __init__(
        self,
        store: EpisodeStore,
        config: SoSnakeConfig | None = None,
        dataset_root: Path = DEFAULT_DATASET_ROOT,
    ) -> None:
        self.store = store
        self.config = config or SoSnakeConfig()
        self.dataset_root = Path(dataset_root)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._progress = ExportProgress()

    # ------------------------------------------------------------------ state

    @property
    def running(self) -> bool:
        with self._lock:
            return self._progress.running

    def progress(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self._progress)
        payload["running"] = payload["phase"] in ("exporting", "verifying")
        return payload

    def _log(self, message: str) -> None:
        with self._lock:
            self._progress.log.append(message)
            # The log is a progress feed, not an archive; the reports carry the
            # detail. Trimming keeps the polled payload bounded on a 50-take run.
            del self._progress.log[:-200]

    # ------------------------------------------------------------------ tasks

    def tasks(self) -> dict[str, Any]:
        """The task labels in the store, with what each would contribute.

        The picker is the one place the operator chooses what a policy will be
        trained on, and the counts are what makes that choice informed: a label
        with four takes under it is not a training set, and it should be visible
        as four rather than as a name in a list.
        """
        metas = self.store.list_meta()
        by_task: dict[str, dict[str, Any]] = {}
        for meta in metas:
            entry = by_task.setdefault(
                meta.task or "", {"task": meta.task or "", "takes": 0, "steps": 0, "seconds": 0.0}
            )
            entry["takes"] += 1
            entry["steps"] += int(meta.n_steps)
            entry["seconds"] += float(meta.duration_s)
        return {
            "tasks": sorted(by_task.values(), key=lambda e: (-e["takes"], e["task"])),
            "dataset_root": str(self.dataset_root),
        }

    # -------------------------------------------------------------- dry run

    def plan(self, config: ExportConfig) -> dict[str, Any]:
        """Screen the selection and report, writing nothing.

        Synchronous: the screening is a second of forward kinematics, not a
        video decode, and a dry run the operator has to poll for is a dry run
        they will skip.
        """
        report, usable = plan(self.store, config, so_snake_config=self.config)
        return {
            "report": _report_payload(report),
            "usable": len(usable),
            "config": _config_payload(config),
        }

    # --------------------------------------------------------------- library

    def datasets(self) -> dict[str, Any]:
        """Every exported dataset under the dataset root, newest first.

        Read from each dataset's own `export.json`, which is the only thing that
        records which takes it was built from -- lerobot's metadata has nowhere
        to put that. A directory without one is listed anyway, marked as not
        ours: it is still a dataset somebody may care about, and silently hiding
        it would be worse than saying it cannot be verified.
        """
        root = self.dataset_root
        if not root.is_dir():
            return {"datasets": [], "root": str(root)}

        found: list[dict[str, Any]] = []
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            entry: dict[str, Any] = {
                "name": path.name,
                "path": str(path),
                "size_bytes": _directory_size(path),
                "modified": _directory_mtime(path),
                "manifest": None,
                "verdict": None,
            }
            try:
                entry["manifest"] = read_manifest(path)
            except (FileNotFoundError, ValueError, OSError):
                entry["manifest"] = None
            entry["verdict"] = self._read_verdict(path)
            found.append(entry)
        found.sort(key=lambda e: e["modified"], reverse=True)
        return {"datasets": found, "root": str(root)}

    def _read_verdict(self, path: Path) -> dict[str, Any] | None:
        """The stored verify result, marked stale if the dataset changed since."""
        file = path / VERDICT_NAME
        try:
            verdict = json.loads(file.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return None
        verdict["stale"] = float(verdict.get("verified_mtime", -1)) < _directory_mtime(path) - 1.0
        return verdict

    def _write_verdict(self, path: Path, report: VerifyReport) -> None:
        payload = _verify_payload(report)
        payload["verified_at"] = time.time()
        payload["verified_mtime"] = _directory_mtime(path)
        try:
            (path / VERDICT_NAME).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            # A read-only dataset directory is not a reason to lose the result;
            # it is already in `_progress` and on its way to the UI.
            pass

    def resolve(self, name_or_path: str) -> Path:
        """A dataset name from the UI back to a path inside the root.

        Names come off the wire, so this refuses anything that would leave the
        root rather than trusting the caller not to send `../`.
        """
        candidate = Path(name_or_path)
        path = candidate if candidate.is_absolute() else (self.dataset_root / candidate)
        path = path.resolve()
        root = self.dataset_root.resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"{name_or_path!r} is not inside {root}")
        if not path.is_dir():
            raise FileNotFoundError(f"no dataset at {path}")
        return path

    # ---------------------------------------------------------------- verify

    def start_verify(self, name_or_path: str) -> dict[str, Any]:
        """Read a dataset back off disk and check it replays. As a job.

        Not a plain request: this decodes every frame of every video to count
        them, which on a full export is minutes. Same slot as the export because
        both are heavy on the same disk and cores, and running them at once
        would only make each slower.
        """
        path = self.resolve(name_or_path)
        with self._lock:
            if self._progress.running:
                raise RuntimeError(
                    f"a {self._progress.kind or 'job'} is already running "
                    f"({self._progress.phase}); stop it before starting another"
                )
            self._stop.clear()
            self._progress = ExportProgress(
                phase="verifying", kind="verify", dataset_path=str(path), repo_id=path.name
            )
            self._thread = threading.Thread(
                target=self._run_verify, args=(path,), name="so-snake-verify", daemon=True
            )
            self._thread.start()
        return self.progress()

    def _run_verify(self, path: Path) -> None:
        try:
            self._log(f"reading {path.name} back to check it replays ...")
            report = verify(path, self.store, so_snake_config=self.config)
            self._write_verdict(path, report)
            with self._lock:
                self._progress.verify_report = _verify_payload(report)
                self._progress.phase = "done" if report.ok else "failed"
                if not report.ok:
                    self._progress.error = report.issues[0]
            self._log(
                "verified: replays back to what was recorded"
                if report.ok
                else f"NOT replayable: {len(report.issues)} problem(s)"
            )
        except Exception as exc:  # noqa: BLE001 - the reason is the payload
            with self._lock:
                self._progress.phase = "failed"
                self._progress.error = f"{type(exc).__name__}: {exc}"
            self._log(f"failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    # ---------------------------------------------------------------- export

    def start(self, config: ExportConfig, *, do_verify: bool = True) -> dict[str, Any]:
        with self._lock:
            if self._progress.running:
                raise RuntimeError(
                    f"an export is already running ({self._progress.phase}); "
                    "stop it before starting another"
                )
            self._stop.clear()
            self._progress = ExportProgress(
                phase="exporting",
                kind="export",
                repo_id=config.repo_id,
                task=config.task or "",
                dataset_path=str(config.root or ""),
            )
            self._thread = threading.Thread(
                target=self._run, args=(config, do_verify), name="so-snake-export", daemon=True
            )
            self._thread.start()
        return self.progress()

    def cancel(self) -> dict[str, Any]:
        self._stop.set()
        return self.progress()

    def _run(self, config: ExportConfig, do_verify: bool) -> None:
        try:
            self._log(f"exporting {config.task or 'every task'} -> {config.repo_id}")
            report = export(
                self.store,
                config,
                so_snake_config=self.config,
                progress=self._on_episode,
                should_continue=lambda: not self._stop.is_set(),
            )
            with self._lock:
                self._progress.report = _report_payload(report)
                self._progress.dataset_path = str(report.dataset_path or "")

            if report.cancelled:
                self._log(
                    f"cancelled after {report.n_episodes} episodes; what was written is "
                    "complete and loadable"
                )
                with self._lock:
                    self._progress.phase = "cancelled"
                return

            self._log(f"wrote {report.n_episodes} episodes / {report.n_frames} frames")

            if not do_verify or report.dataset_path is None:
                with self._lock:
                    self._progress.phase = "done"
                return

            # Verifying by default: the failures that make a dataset unusable --
            # an unwritten parquet footer, a video a frame short, a time axis
            # built from a rate nothing ran at -- all look like success at the
            # moment of writing, and are only found by reading it back.
            with self._lock:
                self._progress.phase = "verifying"
            self._log("reading the dataset back to check it replays ...")
            verified = verify(report.dataset_path, self.store, so_snake_config=self.config)
            self._write_verdict(report.dataset_path, verified)
            with self._lock:
                self._progress.verify_report = _verify_payload(verified)
                self._progress.phase = "done" if verified.ok else "failed"
                if not verified.ok:
                    self._progress.error = verified.issues[0]
            self._log(
                "verified: replays back to what was recorded"
                if verified.ok
                else f"NOT replayable: {len(verified.issues)} problem(s)"
            )
        except Exception as exc:  # noqa: BLE001 - the reason is the payload
            with self._lock:
                self._progress.phase = "failed"
                self._progress.error = f"{type(exc).__name__}: {exc}"
            self._log(f"failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    def _on_episode(self, episode_id: str, frames: int, done: int, total: int) -> None:
        with self._lock:
            self._progress.current_episode = episode_id
            self._progress.episodes_done = done
            self._progress.episodes_total = total
            self._progress.frames_done += frames
        self._log(f"[{done}/{total}] {episode_id}: {frames} frames")


# ------------------------------------------------------------------ payloads


def _config_payload(config: ExportConfig) -> dict[str, Any]:
    return {
        "repo_id": config.repo_id,
        "root": str(config.root or ""),
        "task": config.task,
        "action_space": config.action_space,
        "cameras": list(config.cameras),
        "resolution": list(config.resolution),
        "fps": config.fps,
        "include_aborted": config.include_aborted,
    }


def _report_payload(report: ExportReport) -> dict[str, Any]:
    return {
        "fps": report.fps,
        "action_space": report.action_space,
        "n_episodes": report.n_episodes,
        "n_frames": report.n_frames,
        "cancelled": report.cancelled,
        "dataset_path": str(report.dataset_path or ""),
        "episode_ids": list(report.episode_ids),
        "action_stats": report.action_stats,
        "replay_check": report.replay_check,
        "episodes": [asdict(entry) for entry in report.episodes],
        "skipped": [asdict(entry) for entry in report.skipped],
    }


def _verify_payload(report: VerifyReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["dataset_path"] = str(report.dataset_path)
    payload["ok"] = report.ok
    return payload


def _directory_size(path: Path) -> int:
    """Bytes on disk under `path`. Best effort -- a dataset being written to
    while this walks it reports whatever it saw, which is fine for a size."""
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _directory_mtime(path: Path) -> float:
    """The newest mtime under `path`.

    Not the directory's own: writing a file inside a subdirectory does not touch
    the root's mtime, so a dataset re-exported in place would look untouched and
    a cached verify verdict would stay green over changed bytes.
    """
    newest = 0.0
    for item in path.rglob("*"):
        try:
            newest = max(newest, item.stat().st_mtime)
        except OSError:
            continue
    return newest
