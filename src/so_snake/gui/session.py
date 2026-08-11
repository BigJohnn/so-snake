"""The session manager: one arm, one thing happening to it at a time.

Everything the GUI can do -- teleoperate, record, replay, go home -- ends in
`write_joints_deg` on the same backend, so they cannot overlap. That is the one
invariant this module exists to enforce, and it is enforced by there being a
single `_mode` and a single worker thread rather than by each endpoint checking
the others. A second replay started while teleop is live would fight the loop
for the arm, and on hardware that is not a bug you get to debug afterwards.

Threading, concretely:

  * the worker thread runs the control loop, or the replay, or the homing move;
  * HTTP threads only ever read the snapshot dicts, under `_lock`;
  * stopping is cooperative -- `_stop` is polled between control steps, so the
    loop always finishes the step it is in and the arm is never left holding a
    command that was not also logged;
  * the backend itself is behind `LockedBackend`, because the camera preview
    renders MuJoCo's `mjData` from an HTTP thread while the worker is writing to
    it, and MuJoCo will not survive that unguarded.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import ARM_JOINTS, GRIPPER_JOINT, SoSnakeConfig
from ..data import (
    DEFAULT_EPISODE_ROOT,
    EpisodeRecorder,
    EpisodeReplayer,
    EpisodeStore,
    ReplayConfig,
    ReplayStep,
    inspect_episode,
)
from ..m0_perception import CameraRig, frame_for_preview
from ..m4_execution.motion import move_to_joints
from ..rig import RigSpec, build_backend, build_cameras, build_source
from ..start_pose import (
    DEFAULT_START_POSE_PATH,
    StartPoseError,
    describe_start_pose,
    load_start_pose,
    save_start_pose,
)
from ..teleop.loop import StepRecord, TeleopLoop
from .preview import SimPreview

MODES = ("idle", "teleop", "replay", "homing", "held")

# "held" is homing's resting state, and the reason it exists is torque. Dropping
# torque at the end of a homing move puts the arm on the floor of its own
# workspace -- gravity takes the elbow the moment the servos stop holding -- so
# the pose the operator just asked for is gone before they can teleoperate from
# it. Homing therefore ends *connected and energized*, and the session keeps that
# backend so the next `start_session` adopts it instead of opening a second
# handle on the same serial port. Torque comes off exactly where the operator
# says so: `stop()`.

# How many control steps of history the plots get. 600 at 30 Hz is 20 seconds,
# which is about as far back as an operator looks while judging a take.
SERIES_CAPACITY = 600
EVENT_CAPACITY = 200

# Minimum wall-clock gap between two MuJoCo renders, whatever the browser asks
# for. A render takes tens of milliseconds and holds the backend lock for all of
# them, so an unthrottled preview poll competes directly with the 30 Hz control
# loop -- measured as control steps dropping to 10 Hz while a preview was open.
# Frames within the interval are served from cache, which also means two open
# browser tabs cost one render rather than two.
PREVIEW_MIN_INTERVAL_S = 0.1


class LockedBackend:
    """Serialises access to a backend, so the preview can render it safely.

    Unknown attributes are delegated, and delegation preserves absence: the
    teleoperation loop decides whether to run the mesh clearance check by asking
    `getattr(backend, "command_robot_mesh_min_z_deg", None)`, so a wrapper that
    answered that question for backends which do not have it would turn the
    mock into a crash.
    """

    def __init__(self, backend: Any, lock: threading.Lock) -> None:
        self._backend = backend
        self._lock = lock

    @property
    def inner(self) -> Any:
        return self._backend

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._backend.joint_names

    @property
    def is_connected(self) -> bool:
        return self._backend.is_connected

    def connect(self) -> None:
        with self._lock:
            self._backend.connect()

    def disconnect(self) -> None:
        with self._lock:
            self._backend.disconnect()

    def read_joints_deg(self) -> np.ndarray:
        with self._lock:
            return self._backend.read_joints_deg()

    def write_joints_deg(self, target_deg: np.ndarray) -> None:
        with self._lock:
            self._backend.write_joints_deg(target_deg)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(object.__getattribute__(self, "_backend"), name)
        if callable(attr) and name.startswith("command_"):
            lock = object.__getattribute__(self, "_lock")

            def guarded(*args: Any, **kwargs: Any) -> Any:
                with lock:
                    return attr(*args, **kwargs)

            return guarded
        return attr


@dataclass
class Event:
    """One line of the operator-facing log."""

    time: str
    level: str  # info | warn | error
    message: str

    def to_json(self) -> dict[str, str]:
        return {"time": self.time, "level": self.level, "message": self.message}


def _floats(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, float).ravel()]


class SessionManager:
    """Owns the arm, and whatever is currently driving it."""

    def __init__(
        self,
        config: SoSnakeConfig | None = None,
        episode_root: Path = DEFAULT_EPISODE_ROOT,
        start_pose_path: Path = DEFAULT_START_POSE_PATH,
    ) -> None:
        self.config = config or SoSnakeConfig()
        self.store = EpisodeStore(episode_root)
        self.store.ensure_root()
        # Injectable so a test can record a start pose without rewriting the
        # bench's own; there is one real path and everything ships pointing at it.
        self.start_pose_path = Path(start_pose_path)

        self._lock = threading.RLock()
        self._backend_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        # Keyed by (camera, width, height) rather than a single slot: the page
        # shows both cameras at once, so a one-entry cache would be missed by
        # every request as the two panes alternate, and the throttle it exists
        # to enforce would never bind.
        self._preview: dict[tuple[str, int, int], tuple[float, np.ndarray]] = {}
        self._sim_preview: SimPreview | None = None
        self._sim_preview_model: Any = None
        self._cameras: CameraRig | None = None

        self._mode = "idle"
        self._spec: RigSpec | None = None
        self._backend: LockedBackend | None = None
        self._source: Any = None
        self._loop: TeleopLoop | None = None
        self._recorder: EpisodeRecorder | None = None
        self._replayer: EpisodeReplayer | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error = ""
        self._started_at = 0.0

        # The take batch: how long one episode runs, how many the operator means
        # to collect, how many are done. `_home_pending` is set when a take ends
        # and consumed by the worker on its next control step -- the homing move
        # has to happen on the thread that owns the arm, not on the HTTP thread
        # that pressed the button.
        self._take_steps = 0
        self._take_target = 0
        self._takes_done = 0
        self._home_pending = False
        # The take that just ended by itself, waiting for the operator to say
        # whether it was any good. It is already on disk -- the recorder streams
        # frames and video as they happen, which is what makes a crash keep the
        # take -- so "discard" is a delete, not a decision not to write.
        self._last_take: dict[str, Any] | None = None
        # (mtime_ns, summary) of the recorded start pose; see `_start_pose_summary`.
        self._start_pose_cache: tuple[int, dict[str, Any]] = (-1, {})

        self._latest: dict[str, Any] = {}
        self._series: deque[dict[str, Any]] = deque(maxlen=SERIES_CAPACITY)
        self._events: deque[Event] = deque(maxlen=EVENT_CAPACITY)
        self._replay: dict[str, Any] = _idle_replay()

    # ------------------------------------------------------------------- log

    def log(self, level: str, message: str) -> None:
        with self._lock:
            self._events.append(
                Event(time=datetime.now().strftime("%H:%M:%S"), level=level, message=message)
            )

    # ----------------------------------------------------------------- state

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    @property
    def busy(self) -> bool:
        return self.mode != "idle"

    @property
    def is_held(self) -> bool:
        """Homed, energized, nothing driving it. Busy, but nothing is moving."""
        return self.mode == "held"

    def _require_idle(self, action: str) -> None:
        if self.is_held:
            # Distinct from the generic busy message because the fix is
            # different: nothing is running, the arm is simply still energized,
            # and the operator has to decide whether to release it.
            raise RuntimeError(
                f"cannot {action}: the arm is held at the home pose with torque on. "
                "Start teleop from it, home it again, or press stop to release it."
            )
        if self.busy:
            raise RuntimeError(f"cannot {action}: the arm is busy ({self._mode})")

    def _start_pose_summary(self) -> dict[str, Any]:
        """Where homing will go, re-read only when the file has actually changed.

        A `stat` per poll rather than a parse per poll, and rather than a value
        cached at session start: the operator can record a new start pose from
        the UI, and someone can edit the JSON by hand, and both should show up.
        """
        try:
            stamp = self.start_pose_path.stat().st_mtime_ns
        except OSError:
            stamp = 0
        if self._start_pose_cache[0] != stamp:
            self._start_pose_cache = (stamp, describe_start_pose(self.start_pose_path))
        return self._start_pose_cache[1]

    def status(self) -> dict[str, Any]:
        """One snapshot of everything the UI polls. Cheap; safe to call at 10 Hz."""
        # Read before the lock is taken: it touches the filesystem, and the
        # control loop takes this same lock on every step.
        start_pose = self._start_pose_summary()
        with self._lock:
            spec = self._spec
            return {
                "mode": self._mode,
                "error": self._error,
                "uptime_s": (time.perf_counter() - self._started_at) if self._started_at else 0.0,
                "spec": _spec_json(spec),
                "connected": bool(self._backend is not None and self._backend.is_connected),
                "steps": self._loop.stats.total_steps if self._loop else 0,
                "latest": dict(self._latest),
                # Which cameras are actually open, not which ones were asked
                # for: the difference is the whole point of showing it.
                "cameras": self._cameras.status() if self._cameras else _idle_cameras(),
                "recording": self._recorder.status() if self._recorder else _idle_recording(),
                "start_pose": start_pose,
                "last_take": dict(self._last_take) if self._last_take else _idle_last_take(),
                "takes": {
                    "steps_per_take": self._take_steps,
                    "target_count": self._take_target,
                    "done_count": self._takes_done,
                    "control_hz": self.config.teleop.control_hz,
                },
                "replay": dict(self._replay),
                "events": [event.to_json() for event in list(self._events)[-40:]],
            }

    def series(self, limit: int = SERIES_CAPACITY) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._series)
        return rows[-max(1, limit):]

    # ----------------------------------------------------------------- backend

    def _acquire_backend(self, spec: RigSpec, action: str) -> tuple[LockedBackend, bool]:
        """The backend for `spec`: the held one if it is the same arm, else a new one.

        Adoption is what makes "home, then teleoperate" work on hardware. The
        held backend still owns the serial port, so building a second one fails
        to open it; and even if it could, the first thing a fresh
        `SOFollowerBackend` does not do is keep torque -- the old one would have
        to be disconnected, which drops the arm. So the same object carries over,
        still connected, and teleop starts from the pose the operator homed to.

        Only the identity of the arm has to match. `max_relative_target_deg` is
        applied to the live backend instead, because it is a number the operator
        is entitled to change between homing and teleoperating and re-opening the
        port to honour it would cost the very torque this exists to keep.
        """
        with self._lock:
            held = self._spec if self.is_held else None
            if held is not None and self._backend is not None:
                same_arm = (
                    held.backend == spec.backend
                    and held.port == spec.port
                    and held.robot_id == spec.robot_id
                    and Path(held.joint_map_path) == Path(spec.joint_map_path)
                )
                if not same_arm:
                    raise RuntimeError(
                        f"cannot {action}: the arm is held at home as "
                        f"{held.backend} on {held.port or 'no port'}, and this asks for "
                        f"{spec.backend} on {spec.port or 'no port'}. Press stop to release it first."
                    )
                backend = self._backend
                self._apply_clamp(backend, spec)
                return backend, True

            self._require_idle(action)
            return LockedBackend(build_backend(spec, self.config), self._backend_lock), False

    def _apply_clamp(self, backend: LockedBackend, spec: RigSpec) -> None:
        """Push `spec`'s per-step clamp onto an already-connected backend."""
        setter = getattr(backend.inner, "set_max_relative_target", None)
        if setter is None:  # mock and MuJoCo have no hardware clamp to set
            return
        clamp: dict[str, float] = {j: spec.max_relative_target_deg for j in ARM_JOINTS}
        clamp[GRIPPER_JOINT] = spec.max_relative_target_deg * spec.gripper_speed_mult
        if clamp != backend.inner.max_relative_target:
            setter(clamp)
            self.log("info", f"per-step clamp now {spec.max_relative_target_deg:g} deg (arm)")

    # ------------------------------------------------------------ teleop session

    def start_session(self, spec: RigSpec) -> dict[str, Any]:
        """Bring up a backend and a source, and start teleoperating.

        Started from `held`, this adopts the arm as it stands: torque was never
        dropped, so teleop begins from the home pose rather than from wherever
        gravity left the arm.
        """
        spec.validate()
        backend, adopted = self._acquire_backend(spec, "start a session")
        with self._lock:
            source = build_source(spec, self.config)
            cameras = build_cameras(spec)
            loop = TeleopLoop(source, backend, self.config, stats_capacity=SERIES_CAPACITY)
            recorder = EpisodeRecorder(
                self.store.root,
                config=self.config,
                backend=spec.backend,
                source=spec.source,
                simulated=not spec.is_physical,
                joint_names=backend.joint_names,
                cameras=cameras,
            )

            self._spec, self._backend, self._source = spec, backend, source
            self._loop, self._recorder, self._cameras = loop, recorder, cameras
            self._latest, self._error = {}, ""
            self._series.clear()
            self._mode = "teleop"
            self._stop.clear()
            self._started_at = time.perf_counter()
            # A new session is a new batch of takes; the counts from the last one
            # would otherwise read as progress that has already been made, and
            # its last take is no longer something to pass judgement on.
            self._takes_done, self._home_pending = 0, False
            self._last_take = None

        self.log("info", f"session start: {spec.backend} backend, {spec.source} source"
                         + (" (adopting the held arm)" if adopted else ""))
        self._spawn(self._run_teleop, "so-snake-teleop")
        return self.status()

    def _run_teleop(self) -> None:
        assert self._loop is not None
        try:
            # On the worker, for the same reason the backend and the source
            # connect there: opening two USB cameras takes seconds, and the
            # HTTP request that started the session should not be holding a
            # socket open through it. A failure here reaches the UI through
            # `_spawn`'s handler, the same way a backend that will not connect
            # does.
            cameras = self._cameras
            if cameras is not None and cameras.specs:
                cameras.connect()
                self.log("info", f"cameras open: {', '.join(cameras.roles)}")
            self._loop.run(
                realtime=True,
                on_step=self._on_teleop_step,
                should_continue=lambda: not self._stop.is_set(),
            )
        finally:
            self._teardown("session stopped")

    def _on_teleop_step(self, record: StepRecord) -> None:
        with self._lock:
            self._latest = _telemetry(record)
            self._series.append(_series_row(record))
            recorder = self._recorder
            take_steps = self._take_steps
        if recorder is not None:
            recorder.append(record)
            # A fixed-length take ends itself. Checked after the append so the
            # episode is exactly `take_steps` rows long, and here rather than in
            # the loop because the loop does not know what a take is.
            if take_steps > 0 and recorder.is_recording and recorder.n_steps >= take_steps:
                # `review=True`: nobody decided this one was good -- the clock
                # ended it. It is kept, and offered back for a verdict while the
                # arm walks home, which is the moment the operator actually
                # knows whether the grasp worked.
                self.stop_recording(keep=True, review=True)

        # Set by whichever ended the take -- the length above, or the operator's
        # save/discard on an HTTP thread. Homing runs here because this is the
        # thread that owns the arm, and it blocks the control loop for as long as
        # the move takes, which is correct: nothing should be teleoperating the
        # arm while it walks home.
        if self._home_pending:
            self._home_pending = False
            self._home_between_takes()

    def stop(self) -> dict[str, Any]:
        """Ask the worker to finish its current step and shut down.

        One stop for every mode -- teleop, replay, homing, held -- because there
        is one worker. Blocking until the thread is gone matters: the caller's
        next request may be "start a session on the real arm", and two backends
        holding the same serial port is a mess with no good recovery.

        From `held` there is no worker to wait for: the arm is standing there
        under torque, and stop is the thing that takes the torque off. That is
        why this is the *only* way out of held -- releasing an arm is an action
        someone asks for, never one that happens because a move finished.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10.0)
            if thread.is_alive():
                self.log("error", "worker thread did not stop within 10 s")
        if self.is_held:
            self._teardown("torque released")
        return self.status()

    # -------------------------------------------------------------- recording

    def start_recording(
        self,
        *,
        name: str = "",
        task: str = "",
        notes: str = "",
        steps: int = 0,
        target_count: int = 0,
    ) -> dict[str, Any]:
        """Begin one take.

        `steps` is its length in control frames -- 0 means "until the operator
        says stop", which is what this did before takes had a length. A
        length matters for a dataset: episodes an operator ends by hand vary by
        seconds, and the model sees that variance as signal about the task.

        `target_count` is how many takes the batch is meant to collect. It is a
        target, not a gate: the arm still only records when the operator presses
        the button, so this is the count they are working towards, shown back to
        them, and it does not stop anything when it is reached.
        """
        with self._lock:
            if self._mode == "homing" and self._loop is not None:
                # Between takes. Not an error the operator caused -- the arm is
                # still walking back -- so it says what to wait for.
                raise RuntimeError("the arm is returning home between takes; wait for it to arrive")
            if self._mode != "teleop" or self._recorder is None:
                raise RuntimeError("start a teleop session before recording")
            if self._recorder.is_recording:
                raise RuntimeError("already recording")
            if self._last_take is not None and self._last_take.get("pending"):
                # Starting the next take is a verdict on the last one: the
                # operator moved on, so it stands. Said out loud rather than
                # silently, because the alternative reading is "I forgot".
                self._last_take["pending"] = False
                self.log("info", f"kept {self._last_take['id']} (starting the next take)")
            self._take_steps = max(0, int(steps))
            self._take_target = max(0, int(target_count))
            meta = self._recorder.start(name=name, task=task, notes=notes)
            length = self._take_steps
        hz = self.config.teleop.control_hz
        self.log(
            "info",
            f"recording {meta.id}"
            + (f" — {task}" if task else "")
            + (f" ({length} frames, ~{length / hz:.0f} s)" if length else " (until stopped)"),
        )
        return self.status()

    def stop_recording(self, *, keep: bool = True, review: bool = False) -> dict[str, Any]:
        """End the current take. `review` offers it back for a keep/discard verdict.

        Set for takes that ended on their own: pressing "save" is a decision,
        while running out of frames is not, and the operator has usually only
        just seen how the take went by the time it stops.
        """
        with self._lock:
            recorder = self._recorder
            if recorder is None or not recorder.is_recording:
                raise RuntimeError("no recording in progress")
            episode_id = recorder.episode_id
            steps = recorder.n_steps
        written = recorder.stop(keep=keep)
        with self._lock:
            if written is not None:
                self._takes_done += 1
                self._last_take = {
                    "id": written.id,
                    "name": written.name,
                    "task": written.task,
                    "n_steps": int(written.n_steps),
                    "duration_s": float(written.duration_s),
                    # Only an unreviewed take blocks on a verdict; one the
                    # operator ended by hand has already had one.
                    "pending": bool(review),
                }
            done, target = self._takes_done, self._take_target
            # Home after every take, kept or discarded: the next one should start
            # from the same pose as this one did, and the operator should not
            # have to fly the arm back by hand between takes. Consumed by the
            # worker on its next step; nothing happens if teleop is not running.
            self._home_pending = self._mode == "teleop"
        if written is None:
            self.log("warn", f"discarded {episode_id} ({steps} steps)")
        else:
            progress = f"  [{done}/{target}]" if target else f"  [{done}]"
            self.log(
                "info",
                f"saved {written.id}: {written.n_steps} steps, {written.duration_s:.1f} s{progress}",
            )
            if target and done == target:
                self.log("info", f"batch complete: {done} takes recorded")
        return self.status()

    def decide_last_take(self, *, keep: bool) -> dict[str, Any]:
        """The operator's verdict on the take that just finished.

        `keep` only dismisses the prompt -- the episode is already written, and
        that is deliberate: a take held in memory until someone approved it
        would be a take lost to a crash, an unplugged camera or a full disk,
        which is the one failure this pipeline is built to avoid. Discarding is
        therefore a delete, and it also takes the take back off the batch count,
        so "8 / 10" means eight episodes that are actually worth training on.
        """
        with self._lock:
            take = self._last_take
            if take is None or not take.get("pending"):
                raise RuntimeError("no take is waiting for a verdict")
            episode_id = str(take["id"])
            if keep:
                take["pending"] = False
                self.log("info", f"kept {episode_id}")
                return self.status()

        deleted = self.store.delete(episode_id)
        with self._lock:
            self._last_take = None
            self._takes_done = max(0, self._takes_done - 1)
            done, target = self._takes_done, self._take_target
        progress = f"  [{done}/{target}]" if target else f"  [{done}]"
        self.log(
            "warn" if deleted else "error",
            f"discarded {episode_id}{progress}" if deleted
            else f"could not discard {episode_id}: nothing on disk under that id",
        )
        return self.status()

    # ----------------------------------------------------------------- replay

    def start_replay(
        self,
        episode_id: str,
        spec: RigSpec,
        replay: ReplayConfig,
    ) -> dict[str, Any]:
        """Play an episode back. Refuses if the static inspection finds an error."""
        spec.validate()
        # Adopts a held arm for the same reason teleop does: the replayer walks
        # to the episode's first pose itself, and it should start that walk from
        # the home pose the operator homed to, not from a sagged one.
        backend, _adopted = self._acquire_backend(spec, "start a replay")
        with self._lock:
            episode = self.store.load(episode_id)
            issues = inspect_episode(
                episode,
                self.config,
                backend_joint_names=backend.joint_names,
                target_physical=spec.is_physical,
                replay=replay,
            )
            errors = [i for i in issues if i.level == "error"]
            if errors:
                raise RuntimeError("; ".join(i.message for i in errors))

            replayer = EpisodeReplayer(episode, backend, self.config, replay)
            self._spec, self._backend, self._replayer = spec, backend, replayer
            self._latest, self._error = {}, ""
            self._series.clear()
            self._replay = {
                "active": True,
                "phase": "approach",
                "episode_id": episode_id,
                "mode": replay.mode,
                "speed": replay.speed,
                "step": 0,
                "total": int(episode.meta.n_steps),
                "approach_remaining_deg": 0.0,
                "completed": False,
                "aborted_reason": "",
                "issues": [{"level": i.level, "message": i.message} for i in issues],
                "summary": {},
            }
            self._mode = "replay"
            self._stop.clear()
            self._started_at = time.perf_counter()

        for issue in issues:
            self.log("warn" if issue.level == "warning" else "info", f"replay: {issue.message}")
        self.log("info", f"replay {episode_id} in {replay.mode} mode at {replay.speed:g}x onto {spec.backend}")
        self._spawn(self._run_replay, "so-snake-replay")
        return self.status()

    def _run_replay(self) -> None:
        replayer: EpisodeReplayer = self._replayer
        try:
            report = replayer.run(
                on_step=self._on_replay_step,
                should_continue=lambda: not self._stop.is_set(),
                on_progress=self._on_replay_progress,
            )
            with self._lock:
                self._replay.update(
                    {
                        "phase": "done",
                        "completed": report.completed,
                        "aborted_reason": report.aborted_reason,
                        "summary": {k: float(v) for k, v in report.summary().items()},
                    }
                )
            level = "info" if report.completed else "warn"
            self.log(level, f"replay finished: {report.n_steps} steps"
                            + (f" — {report.aborted_reason}" if report.aborted_reason else ""))
        finally:
            with self._lock:
                self._replay["active"] = False
            self._teardown("replay stopped")

    def _on_replay_step(self, step: ReplayStep) -> None:
        with self._lock:
            self._replay["phase"] = "playing"
            self._replay["step"] = step.index + 1
            self._latest = _replay_telemetry(step)
            self._series.append(
                {
                    "t": round(step.t, 3),
                    "pos_err_mm": round(step.position_error_m * 1000.0, 4),
                    "loop_hz": round(1.0 / step.loop_dt_s, 1) if step.loop_dt_s > 0 else 0.0,
                    "clutch": False,
                    "target": [round(v, 5) for v in step.task_target[:3]],
                    "joints": [round(v, 3) for v in step.commanded_joints_deg],
                    "deviation_deg": round(step.command_deviation_deg, 4),
                    "tracking_deg": round(step.tracking_error_deg, 4),
                }
            )

    def _on_replay_progress(self, phase: str, value: float) -> None:
        with self._lock:
            self._replay["phase"] = phase
            self._replay["approach_remaining_deg"] = round(float(value), 2)

    # ----------------------------------------------------------------- homing

    def start_homing(self, spec: RigSpec) -> dict[str, Any]:
        """Walk the arm to the configured home pose, IK-free and rate-limited.

        Ends in `held`, not `idle`: see the note on MODES. Homing an already-held
        arm is allowed and adopts it, so "home again" costs no torque either.
        """
        spec.validate()
        backend, _adopted = self._acquire_backend(spec, "home the arm")
        with self._lock:
            self._spec, self._backend = spec, backend
            self._error = ""
            self._mode = "homing"
            self._stop.clear()
            self._started_at = time.perf_counter()
        self.log("info", f"homing the {spec.backend} arm")
        self._spawn(self._run_homing, "so-snake-homing")
        return self.status()

    def _home_target(self) -> np.ndarray:
        """Where homing goes: the recorded start pose, else the configured home.

        Read from disk each time rather than cached at session start, so a pose
        recorded mid-session is the one the *next* homing move uses -- which is
        the entire point of being able to record it while teleoperating.

        Falls back rather than failing when the file is unusable: refusing to
        home an energized arm because a JSON file is malformed would leave the
        operator with no way to park it. The reason is logged and shown.
        """
        try:
            recorded = load_start_pose(self.start_pose_path, arm=self.config.arm)
        except StartPoseError as exc:
            self.log("warn", f"{exc}; homing to the configured home pose instead")
            recorded = None
        if recorded is not None:
            return recorded
        gripper_lo, gripper_hi = self.config.arm.joint_limits_deg["gripper"]
        return np.array([*self.config.teleop.home_joints_deg, (gripper_lo + gripper_hi) / 2.0])

    def capture_start_pose(self) -> dict[str, Any]:
        """Record where the arm is right now as the pose homing returns to.

        Meant to be pressed mid-teleoperation: fly the arm to where every take
        should begin, press this, and from then on the automatic homing between
        takes brings it back *there* rather than to the configured default.

        The joints are read from the arm rather than taken from the last
        telemetry frame. It costs one bus read, serialised against the control
        loop by `LockedBackend`, and it is worth it: telemetry carries the arm
        joints and the *commanded* gripper, and a start pose that gets the
        gripper from a different source than the rest of the arm is a pose that
        homes to something nobody looked at.
        """
        with self._lock:
            backend, loop, mode = self._backend, self._loop, self._mode
        if backend is None or mode not in ("teleop", "held") or not backend.is_connected:
            raise RuntimeError(
                "no live arm to read: start a teleop session, or home the arm, before "
                "recording a start pose"
            )

        measured = np.asarray(backend.read_joints_deg(), float)
        task_pose = None
        if loop is not None:
            # The 5D readout is for the operator ("is this inside the box?"),
            # never read back as a target -- homing stays joint-space.
            n_arm = len(self.config.arm.joint_names)
            task_pose = loop.ik.task_pose(measured[:n_arm]).pose.as_array()

        document = save_start_pose(
            measured,
            path=self.start_pose_path,
            arm=self.config.arm,
            limits=self.config.limits,
            task_pose=task_pose,
            source=f"gui/{mode}",
        )
        joints = ", ".join(f"{k} {v:g}" for k, v in document["joints_urdf_deg"].items())
        self.log("info", f"start pose recorded: {joints}")
        if document.get("in_workspace_box") is False:
            # Not an error -- a start pose outside the teleoperation box is a
            # normal thing to want -- but the operator should know that teleop
            # will clamp its way in from there.
            self.log("warn", "start pose is outside the teleoperation workspace box")
        return self.status()

    def _move_home(self, backend: LockedBackend) -> bool:
        return move_to_joints(
            backend,
            self._home_target(),
            step_deg=self.config.teleop.max_joint_step_deg,
            hz=self.config.teleop.control_hz,
            on_progress=lambda remaining: self._on_replay_progress("homing", remaining),
            should_continue=lambda: not self._stop.is_set(),
        )

    def _run_homing(self) -> None:
        backend = self._backend
        try:
            if not backend.is_connected:
                backend.connect()
            reached = self._move_home(backend)
        except Exception:
            # Torque state after a failed move is not something to reason about
            # optimistically: release it and go back to idle, which is what
            # `_spawn`'s handler does with the exception it is about to see.
            self._teardown("homing failed")
            raise
        if self._stop.is_set():
            # The operator asked for the arm to stop while it was on its way.
            # Stop means stop: release it rather than holding it mid-move.
            self._teardown("homing interrupted")
            return
        self.log("info" if reached else "warn",
                 "at home pose" if reached else "homing did not converge")
        self._hold("holding at home — torque on")

    def _hold(self, reason: str) -> None:
        """Park in `held`: connected, energized, nothing driving the arm."""
        with self._lock:
            self._mode = "held"
            self._loop = None
            self._replayer = None
            self._started_at = time.perf_counter()
        self.log("info", reason)

    def _home_between_takes(self) -> None:
        """Walk home mid-session, then hand the arm back to the running loop.

        Called from the control loop's step callback, so the loop is stalled for
        the duration and nothing else is commanding the arm. Afterwards the
        loop's target is re-synced to where the arm now is -- without that it
        would still be holding the target from the end of the take and would
        drag the arm straight back out of the home pose on the next step.
        """
        backend, loop = self._backend, self._loop
        if backend is None or loop is None or self._stop.is_set():
            return
        with self._lock:
            self._mode = "homing"
        self.log("info", "take finished; returning home")
        try:
            reached = self._move_home(backend)
            if not self._stop.is_set():
                self.log("info" if reached else "warn",
                         "at home pose; waiting for the next take"
                         if reached else "homing did not converge; waiting for the next take")
        finally:
            loop.sync_target_to_arm()
            with self._lock:
                if self._mode == "homing":
                    self._mode = "teleop"

    # ------------------------------------------------------------------ preview

    def preview_frame(self, camera: str, width: int, height: int) -> np.ndarray | None:
        """What `camera` currently sees, or None if nothing can answer that.

        A real camera assigned to this role wins over the simulator. That order
        is deliberate rather than alphabetical: the two are never both worth
        showing -- if there is a physical camera pointed at the workspace, a
        render of the model is the less true picture -- and it means the same
        pane, at the same URL, shows the real thing as soon as one is plugged
        in and assigned.

        Neither path can slow the control loop. A camera read is a peek at what
        the capture thread already put down (microseconds, no device I/O). For
        the simulator, only the state copy touches the backend lock; the draw
        runs against `SimPreview`'s own `MjData` on its own GL thread. Frames
        are additionally throttled, which makes two open tabs cost one render
        rather than two.

        None means there is nothing to show at all -- no camera in that role and
        no simulator behind the backend, which is the mock's and the bare real
        arm's situation.
        """
        with self._lock:
            backend = self._backend
            cameras = self._cameras

        key = (camera, width, height)
        now = time.perf_counter()

        with self._preview_lock:
            cached = self._preview.get(key)
            if cached is not None and now - cached[0] < PREVIEW_MIN_INTERVAL_S:
                return cached[1]

            # Throttled before the work, not after: scaling a 1080p frame down
            # and deflating it into a PNG is tens of milliseconds of CPU, and
            # while a take is recording that CPU is wanted by the encoder
            # thread. Two panes polling at 10 Hz would otherwise cost twenty
            # rescales a second for pictures nobody can tell apart.
            frame = self._render_preview(camera, width, height, backend, cameras)
            if frame is None:
                return None
            self._preview[key] = (time.perf_counter(), frame)
            return frame

    def _render_preview(
        self, camera: str, width: int, height: int, backend: Any, cameras: Any
    ) -> np.ndarray | None:
        """Produce one preview frame. Called with `_preview_lock` held."""
        if cameras is not None:
            frame = cameras.read_latest(camera)
            if frame is not None:
                return frame_for_preview(frame, width, height)

        if backend is None:
            return None
        sim = getattr(backend.inner, "sim", None)
        if sim is None:
            return None

        def capture(dest: Any) -> None:
            # Runs on SimPreview's GL thread. The backend lock is held for a
            # memcpy of a few dozen doubles, not for the draw.
            with self._backend_lock:
                dest.qpos[:] = sim.data.qpos
                dest.qvel[:] = sim.data.qvel

        if self._sim_preview is None or self._sim_preview_model is not sim.model:
            if self._sim_preview is not None:
                self._sim_preview.close()
            self._sim_preview = SimPreview(sim.model)
            self._sim_preview_model = sim.model
        return self._sim_preview.frame(camera, width, height, capture)

    # ------------------------------------------------------------------ private

    def _spawn(self, target: Callable[[], None], name: str) -> None:
        def wrapped() -> None:
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - a worker crash must reach the UI
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                self.log("error", self._error)
                traceback.print_exc()
                self._teardown("worker failed")

        thread = threading.Thread(target=wrapped, name=name, daemon=True)
        with self._lock:
            self._thread = thread
        thread.start()

    def _teardown(self, reason: str) -> None:
        """Return to idle: close an open recording, drop the devices, release torque."""
        with self._lock:
            recorder, source, backend = self._recorder, self._source, self._backend
            cameras = self._cameras
            self._cameras = None
            was = self._mode
            self._mode = "idle"
            # The spec goes too. Leaving it set makes the UI show a backend and a
            # source next to "idle", which reads as "loaded and ready" when in
            # fact nothing is connected and the torque is off.
            self._spec = None
            self._loop = None
            self._recorder = None
            self._source = None
            self._backend = None
            self._replayer = None
            self._started_at = 0.0
            self._home_pending = False

        with self._preview_lock:
            # The GL context belongs to the model that is going away with the
            # backend; the next session builds its own.
            if self._sim_preview is not None:
                self._sim_preview.close()
            self._sim_preview, self._sim_preview_model = None, None
            self._preview.clear()

        if recorder is not None and recorder.is_recording:
            # An episode open when the session ends is still evidence -- of a
            # crash, or of an operator who hit stop without hitting stop-record.
            # Keeping it flagged beats silently dropping the take.
            recorder.abort(f"session ended: {reason}")
            written = recorder.stop(keep=True)
            if written is not None:
                self.log("warn", f"session ended mid-recording; kept {written.id}")

        for closer in (
            getattr(cameras, "disconnect", None),
            getattr(source, "disconnect", None),
            getattr(backend, "disconnect", None),
        ):
            if closer is None:
                continue
            try:
                closer()
            except Exception as exc:  # noqa: BLE001 - best effort, keep closing the rest
                self.log("warn", f"error while disconnecting: {type(exc).__name__}: {exc}")

        if was != "idle":
            self.log("info", reason)

    def shutdown(self) -> None:
        """Stop whatever is running. Called when the server is going down."""
        if self.busy:
            self.stop()


# --------------------------------------------------------------------- payloads


def _idle_recording() -> dict[str, Any]:
    return {"recording": False, "id": None, "name": "", "task": "", "steps": 0,
            "duration_s": 0.0, "aborted_reason": ""}


def _idle_last_take() -> dict[str, Any]:
    return {"id": "", "name": "", "task": "", "n_steps": 0, "duration_s": 0.0, "pending": False}


def _idle_cameras() -> dict[str, Any]:
    return {"roles": [], "devices": {}, "connected": []}


def _idle_replay() -> dict[str, Any]:
    return {"active": False, "phase": "idle", "episode_id": "", "mode": "", "speed": 1.0,
            "step": 0, "total": 0, "approach_remaining_deg": 0.0, "completed": False,
            "aborted_reason": "", "issues": [], "summary": {}}


def _spec_json(spec: RigSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {
        "backend": spec.backend,
        "source": spec.source,
        "port": spec.port,
        "physical": spec.is_physical,
        "max_relative_target_deg": spec.max_relative_target_deg,
        "cameras": {camera.role: camera.index_or_path for camera in spec.cameras},
    }


def _telemetry(record: StepRecord) -> dict[str, Any]:
    """The live numbers, rounded to what a screen can show.

    Rounded on the way out rather than in the browser: this crosses the wire ten
    times a second, and `0.010000000000000002` costs more than the precision is
    worth for a readout nobody is measuring from.
    """
    return {
        "index": record.index,
        "t": round(record.t, 3),
        "task_target": [round(v, 5) for v in _floats(record.task_target)],
        "achieved_task_pose": [round(v, 5) for v in _floats(record.achieved_task_pose)],
        "position": [round(v, 5) for v in _floats(record.achieved_position)],
        "commanded_joints_deg": [round(v, 3) for v in _floats(record.commanded_joints_deg)],
        "measured_joints_deg": [round(v, 3) for v in _floats(record.measured_joints_deg)],
        "gripper_cmd_deg": round(record.gripper_cmd_deg, 3),
        "clutch": bool(record.clutch_engaged),
        "loop_hz": round(1.0 / record.loop_dt_s, 1) if record.loop_dt_s > 0 else 0.0,
        "ik_position_error_mm": round(record.ik_position_error_m * 1000.0, 4),
        "ik_pitch_error_deg": round(float(np.degrees(record.ik_pitch_error_rad)), 4),
        "ik_roll_error_deg": round(float(np.degrees(record.ik_roll_error_rad)), 4),
        "robot_mesh_min_z_m": (
            None if record.robot_mesh_min_z_m is None else round(record.robot_mesh_min_z_m, 4)
        ),
        "safety_reason": record.command_safety_reason,
        "flags": {
            "workspace_clamped": bool(record.workspace_clamped),
            "atlas_pitch_clamped": bool(record.atlas_pitch_clamped),
            "atlas_roll_infeasible": bool(record.atlas_roll_infeasible),
            "joint_limit_clamped": bool(record.joint_limit_clamped),
            "joint_rate_clamped": bool(record.joint_rate_clamped),
            "command_safety_held": bool(record.command_safety_held),
            "ik_converged": bool(record.ik_converged),
        },
    }


def _replay_telemetry(step: ReplayStep) -> dict[str, Any]:
    """The same shape as `_telemetry`, so the live panel does not care which is running."""
    return {
        "index": step.index,
        "t": round(step.t, 3),
        "task_target": [round(v, 5) for v in _floats(step.task_target)],
        "achieved_task_pose": [round(v, 5) for v in _floats(step.achieved_task_pose)],
        "position": [round(v, 5) for v in _floats(step.achieved_task_pose[:3])],
        "commanded_joints_deg": [round(v, 3) for v in _floats(step.commanded_joints_deg)],
        "measured_joints_deg": [round(v, 3) for v in _floats(step.measured_joints_deg)],
        "gripper_cmd_deg": round(step.gripper_cmd_deg, 3),
        "clutch": False,
        "loop_hz": round(1.0 / step.loop_dt_s, 1) if step.loop_dt_s > 0 else 0.0,
        "ik_position_error_mm": round(step.position_error_m * 1000.0, 4),
        "ik_pitch_error_deg": round(float(np.degrees(step.pitch_error_rad)), 4),
        "ik_roll_error_deg": round(float(np.degrees(step.roll_error_rad)), 4),
        "robot_mesh_min_z_m": (
            None if step.robot_mesh_min_z_m is None else round(step.robot_mesh_min_z_m, 4)
        ),
        "safety_reason": step.safety_reason,
        "flags": {
            "workspace_clamped": False,
            "atlas_pitch_clamped": False,
            "atlas_roll_infeasible": False,
            "joint_limit_clamped": bool(step.limit_clamped),
            "joint_rate_clamped": bool(step.rate_clamped),
            "command_safety_held": bool(step.safety_held),
            "ik_converged": bool(step.ik_converged),
        },
    }


def _series_row(record: StepRecord) -> dict[str, Any]:
    return {
        "t": round(record.t, 3),
        "pos_err_mm": round(record.ik_position_error_m * 1000.0, 4),
        "loop_hz": round(1.0 / record.loop_dt_s, 1) if record.loop_dt_s > 0 else 0.0,
        "clutch": bool(record.clutch_engaged),
        "target": [round(v, 5) for v in _floats(record.task_target[:3])],
        "joints": [round(v, 3) for v in _floats(record.commanded_joints_deg)],
        "deviation_deg": 0.0,
        "tracking_deg": round(
            float(np.abs(np.asarray(record.commanded_joints_deg) - np.asarray(record.measured_joints_deg)).max()),
            4,
        ),
    }
