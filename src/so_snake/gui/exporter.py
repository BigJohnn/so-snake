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

# Bumped whenever the verdict payload changes shape or a check changes meaning.
#
# A cached verdict is an answer computed by a particular version of `verify`.
# When that code changes, the old answer is not merely missing a field -- it is
# a claim made by checks that no longer exist in the form they were run. So a
# verdict written under a different version is discarded rather than migrated,
# and the dataset reads as "not verified yet". The cost is one re-verify; the
# alternative is a green badge attesting to a check that was never run. (This
# already bit once: `skipped` was added to the report, and verdicts written
# before it lacked the field that decides OK from PARTIAL.)
#
# 2: a source take missing from the store moved from `issues` to `skipped` (see
#    `verify`), and `episodes_compared` was added. Verdicts written under 1 say
#    FAILED where this version says PARTIAL, which is precisely the kind of stale
#    claim this counter exists to discard.
VERDICT_VERSION = 2


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
                meta.task or "", {"task": meta.task or "", "takes": 0, "steps": 0, "seconds": 0.0,
                                 # The ROI editor uses a real recorded frame,
                                 # not a live camera, so the export contract is
                                 # chosen from the evidence it will transform.
                                 "sample_episode": meta.id}
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

        The entry for each dataset carries the manifest when it exists, and a
        synthesized equivalent when it does not. The synthesized manifest comes
        from lerobot's own `meta/info.json` -- enough for the GUI to know the
        episode count, fps and action space (so replay can pick an episode)
        without claiming a source mapping we do not have. `episode_ids` is the
        single thing the manifest uniquely contributes, and it stays empty for
        synthesized ones; the GUI uses that to decide whether to show
        "原始 take"折叠面板.
        """
        from ..data.export import dataset_meta

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
                # `ours` is the GUI's hook for "this dataset has a manifest we
                # wrote, so source-fidelity is checkable". Synthesized
                # manifests get `false`; the GUI uses this to gate the "原始
                # take"折叠面板 rather than inferring from `episode_ids`.
                "ours": False,
                "verdict": None,
            }
            try:
                manifest, ours = dataset_meta(path)
                entry["manifest"] = manifest
                entry["ours"] = ours
            except (FileNotFoundError, ValueError, OSError):
                entry["manifest"] = None
                entry["ours"] = False
            entry["verdict"] = self._read_verdict(path)
            found.append(entry)
        found.sort(key=lambda e: e["modified"], reverse=True)
        return {"datasets": found, "root": str(root)}

    def _read_verdict(self, path: Path) -> dict[str, Any] | None:
        """The stored verify result, or None if it cannot be trusted.

        None covers three cases that all mean the same thing to the operator --
        "this has not been verified by the code now running": no file, an
        unreadable one, and one written under a different `VERDICT_VERSION`.
        `stale` is the softer case: the verdict is this version's, but the
        dataset has been written to since, so the answer is about older bytes.

        `dataset_mtime` rides along because the UI has to say *when* the dataset
        changed, and the verdict alone cannot know that -- it only carries the
        mtime as of its own run. Both are added here rather than stored, so they
        describe the directory as it is now.
        """
        file = path / VERDICT_NAME
        try:
            verdict = json.loads(file.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return None
        if not isinstance(verdict, dict) or verdict.get("version") != VERDICT_VERSION:
            return None
        mtime = _directory_mtime(path)
        verdict["dataset_mtime"] = mtime
        verdict["stale"] = float(verdict.get("verified_mtime", -1)) < mtime - 1.0
        return verdict

    def _write_verdict(self, path: Path, report: VerifyReport) -> None:
        payload = _verify_payload(report)
        payload["version"] = VERDICT_VERSION
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
        """A dataset name from the UI back to a path *inside* the root.

        Names come off the wire, so this refuses anything that would leave the
        root rather than trusting the caller not to send `../`.

        It also refuses the root itself, which is not a nicety. `data/lerobot`
        is a container of datasets, and `Path("")` and `Path(".")` both join to
        it -- so a caller that omitted the field, or sent an undefined one that
        JSON dropped, used to get the container back and have it opened as a
        dataset. The failure then surfaced three layers down as "meta/info.json
        is missing", which describes the root accurately and explains nothing.
        A missing name is a missing name, and it should say so here.
        """
        name = str(name_or_path).strip()
        root = self.dataset_root.resolve()
        if not name or name in (".", "/"):
            raise ValueError(
                "no dataset was named. Pick one from the library -- "
                f"{root} is the directory they live in, not a dataset itself"
            )

        candidate = Path(name)
        path = (candidate if candidate.is_absolute() else (self.dataset_root / candidate)).resolve()
        if path == root:
            raise ValueError(
                f"{root} is the dataset root, not a dataset. Name one of the "
                "directories inside it"
            )
        if root not in path.parents:
            raise ValueError(f"{name!r} is not inside {root}")
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

    def start(
        self,
        config: ExportConfig,
        *,
        do_verify: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Any]:
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
                target=self._run,
                args=(config, do_verify, overwrite),
                name="so-snake-export",
                daemon=True,
            )
            self._thread.start()
        return self.progress()

    def cancel(self) -> dict[str, Any]:
        self._stop.set()
        return self.progress()

    def _run(self, config: ExportConfig, do_verify: bool, overwrite: bool) -> None:
        try:
            self._log(f"exporting {config.task or 'every task'} -> {config.repo_id}")
            report = export(
                self.store,
                config,
                so_snake_config=self.config,
                progress=self._on_episode,
                should_continue=lambda: not self._stop.is_set(),
                overwrite=overwrite,
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
    # `skipped` may not exist on older VerifyReport definitions; defaulting
    # here keeps the gateway honest if someone reverts that field.
    payload.setdefault("skipped", [])
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
    """The newest mtime under `path`, ignoring the verdict file.

    Not the directory's own: writing a file inside a subdirectory does not touch
    the root's mtime, so a dataset re-exported in place would look untouched and
    a cached verify verdict would stay green over changed bytes.

    The verdict file is excluded because it is *about* the dataset rather than
    part of it, and counting it made the staleness check self-defeating: the
    mtime recorded inside `verify.json` is read before the file is written, so
    the write itself became the newest thing under the directory and every
    verdict read back as "the dataset changed after it was verified" -- within
    seconds of being computed. Excluding it also keeps verifying a dataset from
    reordering the library, which sorts on this number.
    """
    verdict = path / VERDICT_NAME
    newest = 0.0
    for item in path.rglob("*"):
        if item == verdict:
            continue
        try:
            newest = max(newest, item.stat().st_mtime)
        except OSError:
            continue
    return newest
