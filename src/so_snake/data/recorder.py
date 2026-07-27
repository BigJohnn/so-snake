"""Recording: turning a live teleoperation loop into episodes on disk.

The recorder is a *listener*. It hangs off `TeleopLoop.run(on_step=...)` and
never influences what the loop commands, which is the whole point: an episode is
evidence of what the controller did, and a recorder that could change the
control path would be recording itself as much as the robot.

Episode boundaries are the operator's, not the loop's. One teleop session
usually yields many demonstrations with re-setting of the scene in between, so
`start()` / `stop()` are separate from the session lifecycle and can be called
repeatedly against one running loop.

Steps are buffered in memory and written when the episode ends. A demonstration
is seconds to a couple of minutes -- a few thousand rows of a few hundred bytes
-- so the buffer is small, and writing once means an episode directory is either
complete or absent, never a truncated file that looks fine until training. The
`max_steps` cap is the backstop for a recording someone forgot to stop.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import SoSnakeConfig
from ..teleop.loop import LoopStats, StepRecord
from .episode import (
    DEFAULT_EPISODE_ROOT,
    EpisodeMeta,
    encode_frames,
    utc_now_iso,
    write_episode,
)

# 30 minutes at 30 Hz. Anything longer is a session, not a demonstration.
DEFAULT_MAX_STEPS = 54_000


def config_snapshot(config: SoSnakeConfig) -> dict[str, Any]:
    """The tuning that was in force, as plain JSON.

    Everything that shapes the recorded action space: the workspace box, the
    rate caps, the home pose, the IK tolerances. Paths are stringified rather
    than dropped -- which URDF produced these joint angles is part of the
    episode's meaning.
    """

    def plain(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    return plain(asdict(config))


class EpisodeRecorder:
    """Buffers `StepRecord`s into episodes and writes them out.

    Safe to `start`/`stop` from one thread while `append` is called from the
    control loop's thread, which is exactly how the GUI drives it.
    """

    def __init__(
        self,
        root: Path = DEFAULT_EPISODE_ROOT,
        *,
        config: SoSnakeConfig | None = None,
        backend: str = "",
        source: str = "",
        simulated: bool = True,
        joint_names: tuple[str, ...] = (),
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.root = Path(root)
        self.config = config or SoSnakeConfig()
        self.backend = backend
        self.source = source
        self.simulated = simulated
        self.joint_names = tuple(joint_names)
        self.max_steps = int(max_steps)

        self._lock = threading.Lock()
        self._records: list[StepRecord] = []
        self._meta: EpisodeMeta | None = None
        self._aborted_reason = ""

    # ------------------------------------------------------------------ state

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._meta is not None

    @property
    def episode_id(self) -> str | None:
        with self._lock:
            return self._meta.id if self._meta else None

    @property
    def n_steps(self) -> int:
        with self._lock:
            return len(self._records)

    def status(self) -> dict[str, Any]:
        """A snapshot for the GUI. Cheap enough to poll."""
        with self._lock:
            if self._meta is None:
                return {"recording": False, "id": None, "name": "", "task": "", "steps": 0,
                        "duration_s": 0.0, "aborted_reason": self._aborted_reason}
            duration = self._records[-1].t - self._records[0].t if len(self._records) > 1 else 0.0
            return {
                "recording": True,
                "id": self._meta.id,
                "name": self._meta.name,
                "task": self._meta.task,
                "steps": len(self._records),
                "duration_s": float(duration),
                "aborted_reason": self._aborted_reason,
            }

    # ------------------------------------------------------------- life cycle

    def start(self, *, name: str = "", task: str = "", notes: str = "") -> EpisodeMeta:
        """Open a new episode. Any episode already open is discarded, not saved.

        Discarding is the right default for a double-press: the operator meant
        to begin a take, and silently saving the abandoned half as a demo would
        put an unlabelled fragment into the training set.
        """
        with self._lock:
            self._records = []
            self._aborted_reason = ""
            meta = EpisodeMeta(
                id=self._new_id(),
                name=name,
                task=task,
                notes=notes,
                created_at=utc_now_iso(),
                backend=self.backend,
                source=self.source,
                simulated=self.simulated,
                control_hz=float(self.config.teleop.control_hz),
                joint_names=list(self.joint_names),
                config=config_snapshot(self.config),
            )
            self._meta = meta
            return meta

    def append(self, record: StepRecord) -> None:
        """Called once per control step. Never raises into the control loop."""
        with self._lock:
            if self._meta is None:
                return
            if len(self._records) >= self.max_steps:
                self._aborted_reason = f"step cap reached ({self.max_steps})"
                return
            self._records.append(record)

    def abort(self, reason: str) -> None:
        """Flag the open episode as having ended badly. Does not close it."""
        with self._lock:
            if self._meta is not None:
                self._aborted_reason = reason

    def stop(self, *, keep: bool = True) -> EpisodeMeta | None:
        """Close the open episode. Returns its meta if it was written.

        `keep=False` throws the take away -- the operator fumbled it, the cube
        was in the wrong place, the clutch was never pressed. Discarding at the
        moment they know is far better than filtering later from a spreadsheet.
        """
        with self._lock:
            meta, records = self._meta, self._records
            reason = self._aborted_reason
            self._meta, self._records, self._aborted_reason = None, [], ""

        if meta is None:
            return None
        if not keep or not records:
            return None

        stats = LoopStats()
        for record in records:
            stats.add(record)

        meta.n_steps = len(records)
        meta.duration_s = float(records[-1].t - records[0].t) if len(records) > 1 else 0.0
        meta.summary = {k: float(v) for k, v in stats.summary().items()}
        meta.aborted_reason = reason
        write_episode(self.root, meta, encode_frames(records))
        return meta

    # ----------------------------------------------------------------- naming

    def _new_id(self) -> str:
        """`ep_YYYYmmdd_HHMMSS`, with a numeric suffix if that second is taken."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = f"ep_{stamp}"
        n = 1
        while (self.root / candidate).exists():
            n += 1
            candidate = f"ep_{stamp}_{n}"
        return candidate
