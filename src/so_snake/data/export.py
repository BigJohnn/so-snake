"""Turning recorded episodes into a `LeRobotDataset` that trains a 5D policy.

The recorded format is deliberately redundant (see `episode.py`); a training set
is the opposite -- it has to commit to one state, one action, one frame rate.
This module is where those commitments are made, and each one is a decision that
the recorded data forced rather than a default:

## State is the pose the arm reached, not the pose it was told to reach

`observation.state.task_pose` reads like the observation and is not one. It is
the forward kinematics of the *IK solution*, so its distance to
`action.task.target` is the solver residual -- measured at 1e-6 across this
bench's takes. The arm's real distance from its target is three orders of
magnitude larger: median 9.6 mm, p95 41 mm, p95 pitch 10 degrees, all of it
servo lag under load. Training a policy on FK(command) would hand it a
proprioceptive channel that cannot report a stalled joint, a dropped object or a
collision, because none of those change what was commanded.

So the state exported here is FK(`observation.state.joints_deg`) -- recomputed
at export time from the joint angles the bus actually returned.

## Action is a step from that reached pose, not from the previous command

The map is **absolute pose on the 5D manifold -> delta in the same 5D chart**.
Both sides are the one chart `(x, y, z, pitch, roll)` from `m3_safety.task_pose`;
no SE(3), no quaternion, no rotation matrix reaches the dataset.

With `q_t` the joints the bus reported, `Phi` the forward map into the chart
(`TaskIK5D.task_pose`), `p_t = Phi(q_t)` the pose reached and `c_t` the 5D
target the loop commanded:

    state[t]  = ( p_t,               gripper measured )
    action[t] = ( c_t (-) p_{t-1},   gripper commanded )

and the update formula -- the inverse, run once per step at rollout:

    c_t = p_{t-1} (+) action[t][0:5]

where `(+)` and `(-)` are componentwise and differ from plain arithmetic only on
the two angular coordinates, which are wrapped into (-pi, pi]:

    x, y, z       a_i +/- b_i                  metres
    pitch, roll   wrap(a_i +/- b_i)            radians

`apply_action` below *is* `(+)`, and it is the only implementation of it, so
training and rollout cannot drift apart. At rollout `p_{t-1}` is measured on the
spot rather than read back:

    c_t = Phi(q_measured) (+) policy(observation)[0:5]

which is the whole point of the anchor. An increment has to be anchored to
something, and the choice decides whether errors accumulate. Anchored to the
previous *target*, the policy is an open-loop integrator: nothing in the loop
ever compares it to the arm, so a systematic under-prediction walks away and
never comes back. Anchored to the *reached* pose, every step re-references the
measurement, and a rollout corrects itself.

Both sides then contain the same standing servo lag, so the policy reproduces
the lead the teleoperator's commands carried instead of trailing it. The lag
being larger than one step's motion is the reason this matters here: at p95, the
delta from lag is 41 mm against 5 mm of operator intent, so an anchor that
ignores it is not a small approximation.

`(-)` is a plain difference of chart coordinates, not a Lie-algebra log. That is
sound because the chart is regular everywhere the arm can reach -- its azimuth
is a function of *position*, not of the tool's own heading, specifically so it
does not go singular when the gripper points straight down. Over one step the
motion is small and the chart is smooth, so the coordinate difference and the
geodesic step agree to far inside the servo lag.

`--action-space absolute` exports `target[t]` instead. It is the control: if a
delta rollout drifts, the absolute model says whether the drift came from the
action space or from somewhere else.

## The gripper stays absolute in both

It is a two-state signal in practice -- these takes only ever hold 2 deg or
90 deg and pass through the middle -- and a delta on it would spend the whole
episode predicting zero and then have to hit a 88 degree jump exactly once. As
state it is the *measured* angle, which is the only channel that reports the
jaws stalling on an object; format v1 episodes never stored it and fall back to
the commanded angle, which the export report calls out per episode.

## One frame rate, taken from the clock rather than the config

The loop is configured for 30 Hz and achieves 26.1. Recording wrote the
configured rate into the mp4 headers, so the videos claim to be 15% shorter than
the takes they came from. Exporting at the configured rate would train a policy
whose action chunk spans 3.3 s of intent, then replay it over 2.9 s of wall
clock -- a rollout 15% faster than every demonstration it learned from, on an
arm whose tracking lag is already the largest term in the action.

The rate here is therefore measured: `n_steps / duration_s` per episode, rounded
across the selection. Frames are read out of the videos by index and re-encoded
at that rate, which is also what removes the header lie -- lerobot seeks these
files by timestamp, so a header disagreeing with the row grid is not cosmetic.

(The recording loop now holds its configured rate -- see `so_snake.pacing` --
so takes recorded after that fix measure 30 Hz and the two agree. Measuring is
still what decides it, because the 43 takes recorded before it are at 26 and
mixing the two on one time grid is exactly what the screening has to catch.)

## What is written alongside the dataset, and why

`export.json` in the dataset root records the config and the ordered list of
source episode ids. lerobot's own metadata cannot say which take a dataset
episode came from, and without that the export is a one-way door: nothing can
check the rows on disk against what they were made from, re-export a corrected
subset, or tell an operator which take to go and re-record. `verify` reads it
back and does exactly that comparison -- against the parquet on disk, not
against the arrays that were in memory when the parquet was written, because
the failures worth catching (a missing footer, a dropped video frame, a
timestamp grid that disagrees with the rows) all live in the gap between them.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from ..config import SoSnakeConfig
from ..m3_safety.task_pose import SO100TaskPose, wrap_to_pi
from .episode import Episode, EpisodeMeta
from .store import EpisodeStore

TASK_DIM_NAMES: tuple[str, ...] = ("x", "y", "z", "pitch", "roll")

# The export's own record of what it made, next to lerobot's metadata rather
# than inside it: lerobot owns `meta/`, and a file this repository writes into
# a directory another library manages should be one that library will never
# look at.
MANIFEST_NAME = "export.json"

STATE_DIM = 6  # (x, y, z, pitch, roll, gripper)
ACTION_DIM = 6

ACTION_SPACES = ("delta", "absolute")

# The two angular components of the 5D manifold. A difference of angles is only
# a tangent step once it is wrapped -- roll is limited to +-180 deg, so a take
# that crosses the seam would otherwise contribute a 2*pi action that no policy
# should ever be asked to reproduce.
_ANGULAR_DIMS = (3, 4)


@dataclass
class ExportConfig:
    """What to select, how to shape it, and where to put it."""

    repo_id: str
    root: Path | None = None

    # Exact `meta.task` string to select. None takes every episode in the store,
    # which is almost never what you want: the tasks in one store are different
    # skills, and a policy trained across them learns their average.
    task: str | None = None
    episode_ids: tuple[str, ...] = ()

    action_space: str = "delta"
    cameras: tuple[str, ...] = ("third_person", "wrist")
    resolution: tuple[int, int] = (240, 320)  # (height, width)

    # None measures the rate from the takes. An explicit value is honoured and
    # reported against the measurement, because overriding it is exactly the
    # thing that makes a rollout run at the wrong speed.
    fps: int | None = None

    # An episode whose measured rate is this far from the dataset rate is
    # rejected: it cannot share a single-integer timeline with the others.
    fps_tolerance: float = 0.08

    include_aborted: bool = False

    # Only recorded into the manifest, so a dataset can say which store it came
    # from. The export reads episodes through the `EpisodeStore` it is handed.
    episode_root: Path | None = None

    def __post_init__(self) -> None:
        if self.action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space must be one of {ACTION_SPACES}, got {self.action_space!r}"
            )
        if not self.cameras:
            raise ValueError("at least one camera is required")
        height, width = self.resolution
        if height <= 0 or width <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")


@dataclass
class EpisodeReport:
    """What the exporter decided about one episode, and why."""

    episode_id: str
    task: str
    n_steps: int
    measured_fps: float
    gripper_measured: bool
    included: bool
    reason: str = ""
    # What the loop was configured for when this take was recorded. Kept beside
    # the measurement because the gap between them is the one thing that
    # explains a dataset rate the operator did not expect -- and it is a
    # property of the recording, not of the export, so no amount of re-exporting
    # will change it.
    configured_hz: float = 0.0


@dataclass
class ExportReport:
    fps: int
    action_space: str
    n_episodes: int = 0
    n_frames: int = 0
    episodes: list[EpisodeReport] = field(default_factory=list)
    action_stats: dict[str, Any] = field(default_factory=dict)
    replay_check: dict[str, Any] = field(default_factory=dict)
    dataset_path: Path | None = None
    # Source episode ids in dataset episode order; see `write_manifest`.
    episode_ids: tuple[str, ...] = ()
    cancelled: bool = False

    @property
    def skipped(self) -> list[EpisodeReport]:
        return [e for e in self.episodes if not e.included]


def measured_fps(episode: Episode) -> float:
    """The rate the loop actually held, from the step count and the clock."""
    return episode.playback_hz


def observed_task_pose(episode: Episode, config: SoSnakeConfig | None = None) -> np.ndarray:
    """`(n, 5)` the pose the arm reached, from the joints the bus reported.

    Recomputed rather than read: the recorded `observation.state.task_pose` is
    FK of the IK solution, not of the measurement. See the module docstring.
    """
    from ..m3_safety.ik5d import TaskIK5D

    config = config or SoSnakeConfig()
    ik = TaskIK5D(arm=config.arm, teleop=config.teleop, ik=config.ik)
    joints = episode.column("observation.state.joints_deg")
    return np.array([ik.task_pose(q).pose.as_array() for q in joints], dtype=np.float32)


def build_state_action(
    episode: Episode,
    action_space: str,
    config: SoSnakeConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """`(state, action, gripper_was_measured)` for one episode.

    state  `(n, 6)` = reached (x, y, z, pitch, roll) + gripper angle
    action `(n, 6)` = manifold step or absolute target, + absolute gripper
    """
    pose = observed_task_pose(episode, config)
    target = np.asarray(episode.task_target, dtype=np.float32)
    gripper_state, gripper_measured = episode.measured_gripper_deg()
    gripper_cmd = np.asarray(episode.gripper_cmd_deg, dtype=np.float32)

    state = np.concatenate(
        [pose, np.asarray(gripper_state, dtype=np.float32).reshape(-1, 1)], axis=1
    )

    if action_space == "absolute":
        task_action = target.copy()
    else:
        # Anchor each step on the pose reached at the *previous* step, which is
        # what the rollout will have in hand when it asks for this action. The
        # first row has no predecessor and anchors on its own pose, so it
        # carries the standing servo lag with no operator motion in it -- a few
        # millimetres, and one row out of several hundred. It is not zero, and
        # a rollout starting from rest reproduces the same thing.
        previous = np.vstack([pose[:1], pose[:-1]])
        task_action = target - previous
        for dim in _ANGULAR_DIMS:
            task_action[:, dim] = wrap_to_pi(task_action[:, dim])

    action = np.concatenate([task_action, gripper_cmd.reshape(-1, 1)], axis=1)
    return state.astype(np.float32), action.astype(np.float32), gripper_measured


def replay_targets_from_state_action(
    state: np.ndarray, action: np.ndarray, action_space: str
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the 5D target stream from exported rows.

    This is the offline replay contract for the dataset. A learned policy uses
    the same inverse one step at a time against the arm's current measurement;
    a recorded exported episode can be replayed without the arm by anchoring row
    i on exported state i-1, because that is the measurement the operator had
    just produced when row i's command was generated.
    """
    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"expected state shape (n, {STATE_DIM}), got {state.shape}")
    if action.ndim != 2 or action.shape != state.shape:
        raise ValueError(f"expected action shape {state.shape}, got {action.shape}")

    targets: list[np.ndarray] = []
    gripper: list[float] = []
    for i in range(len(state)):
        anchor = state[i, :5] if action_space == "absolute" or i == 0 else state[i - 1, :5]
        target, grip = apply_action(anchor, action[i], action_space)
        targets.append(target)
        gripper.append(grip)
    return np.asarray(targets, dtype=np.float32), np.asarray(gripper, dtype=np.float32)


def decode_video(path: Path, count: int, resolution: tuple[int, int]) -> Iterator[np.ndarray]:
    """Yield exactly `count` RGB frames, resized, in recording order.

    By index, never by timestamp. The recorder writes one frame per control step
    so that video frame *i* is row *i*; the file's own timeline was built from
    the configured rate and disagrees with the wall clock, so seeking by time
    here would reintroduce the very skew this export exists to remove.
    """
    import av

    height, width = resolution
    produced = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if produced >= count:
                break
            yield frame.reformat(width=width, height=height, format="rgb24").to_ndarray()
            produced += 1
    if produced != count:
        raise ValueError(
            f"{path.name} decoded {produced} frames, expected {count}; the video and "
            "the frame table are no longer aligned and this episode cannot be exported"
        )


def _video_frame_count(episode: Episode, role: str) -> int | None:
    cameras = episode.meta.video.get("cameras", {}) if episode.meta.video else {}
    entry = cameras.get(role)
    if not entry:
        return None
    return int(entry.get("written", 0))


def _screen(episode: Episode, config: ExportConfig, fps: int | None) -> str:
    """Return the reason this episode cannot be exported, or an empty string."""
    if episode.meta.aborted_reason and not config.include_aborted:
        return f"aborted: {episode.meta.aborted_reason}"
    if episode.meta.n_steps <= 0:
        return "no steps"
    for role in config.cameras:
        written = _video_frame_count(episode, role)
        if written is None:
            return f"no {role} camera"
        if written != episode.meta.n_steps:
            return f"{role} wrote {written} frames for {episode.meta.n_steps} steps"
        if not (episode.path / f"{role}.mp4").is_file():
            return f"{role}.mp4 missing"
    if fps is not None:
        rate = measured_fps(episode)
        if abs(rate - fps) / fps > config.fps_tolerance:
            return f"ran at {rate:.2f} Hz, dataset is {fps} Hz"
    return ""


def select_episodes(store: EpisodeStore, config: ExportConfig) -> list[Episode]:
    """Load the episodes the config asks for, newest last so ids stay ordered."""
    if config.episode_ids:
        chosen = [store.load(eid) for eid in config.episode_ids]
    else:
        metas = [m for m in store.list_meta() if config.task is None or m.task == config.task]
        chosen = [store.load(m.id) for m in metas]
    chosen.sort(key=lambda e: (e.meta.created_at, e.meta.id))
    return chosen


def resolve_fps(episodes: list[Episode], config: ExportConfig) -> int:
    """One integer rate for the whole dataset, measured unless overridden.

    The median, not the mean: a single take that ran at a wildly different rate
    -- a stall, a take recorded headless as fast as the mock would go -- should
    be rejected by the screening, not allowed to drag the grid that every other
    take is then measured against.
    """
    if config.fps is not None:
        return int(config.fps)
    if not episodes:
        raise ValueError("no episodes to measure a frame rate from")
    rates = [measured_fps(e) for e in episodes]
    return max(1, int(round(float(np.median(rates)))))


def _features(config: ExportConfig) -> dict[str, dict[str, Any]]:
    height, width = config.resolution
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [*TASK_DIM_NAMES, "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": [
                *(
                    TASK_DIM_NAMES
                    if config.action_space == "absolute"
                    else [f"d{name}" for name in TASK_DIM_NAMES]
                ),
                "gripper",
            ],
        },
    }
    for role in config.cameras:
        features[f"observation.images.{role}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def plan(
    store: EpisodeStore,
    config: ExportConfig,
    *,
    so_snake_config: SoSnakeConfig | None = None,
) -> tuple[ExportReport, list[tuple[Episode, np.ndarray, np.ndarray]]]:
    """Everything the export decides before it touches a video or a dataset.

    Shared with `--dry-run`, which is the whole point: a dry run that screened
    episodes by different rules than the real export would be worthless as a
    pre-flight.
    """
    episodes = select_episodes(store, config)
    if not episodes:
        raise ValueError(
            "no episodes matched"
            + (f" task {config.task!r}" if config.task else "")
            + "; nothing to export"
        )

    fps = resolve_fps(episodes, config)
    report = ExportReport(fps=fps, action_space=config.action_space)

    usable: list[tuple[Episode, np.ndarray, np.ndarray]] = []
    for episode in episodes:
        reason = _screen(episode, config, fps)
        entry = EpisodeReport(
            episode_id=episode.meta.id,
            task=episode.meta.task,
            n_steps=int(episode.meta.n_steps),
            measured_fps=measured_fps(episode),
            gripper_measured=False,
            included=not reason,
            reason=reason,
            configured_hz=float(episode.meta.control_hz),
        )
        if not reason:
            state, action, entry.gripper_measured = build_state_action(
                episode, config.action_space, so_snake_config
            )
            usable.append((episode, state, action))
            report.n_episodes += 1
            report.n_frames += len(state)
        report.episodes.append(entry)

    if usable:
        report.action_stats = _action_stats(np.concatenate([a for _, _, a in usable]))
        report.replay_check = _replay_check(usable, config.action_space)
    return report, usable


def export(
    store: EpisodeStore,
    config: ExportConfig,
    *,
    so_snake_config: SoSnakeConfig | None = None,
    progress: Callable[[str, int, int, int], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    overwrite: bool = False,
) -> ExportReport:
    """Write the selected episodes into a `LeRobotDataset` and report on it.

    `progress(episode_id, frames, done, total)` is called after each episode.
    `should_continue` is polled between episodes -- between, not within, so a
    cancelled export leaves whole episodes behind rather than half of one.

    The target directory is owned by whoever is on the other end of this call.
    If it already has files in it, this function refuses with `FileExistsError`
    unless `overwrite=True` -- silent replacement of a dataset that an operator
    was already using is a worse failure than refusing to start. The error
    message names what's there, so the operator can decide between `--overwrite`
    and a fresh directory.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    report, usable = plan(store, config, so_snake_config=so_snake_config)
    if not usable:
        raise ValueError(
            "every matched episode was rejected:\n"
            + "\n".join(f"  {e.episode_id}: {e.reason}" for e in report.skipped)
        )
    fps = report.fps

    root = Path(config.root)
    if root.exists():
        # Detect what is there so the operator's choice is informed: "replace
        # this 10-episode / 4221-frame dataset" is a different decision from
        # "replace this 4 KB stub from a half-deleted prior run".
        existing_summary = _summarise_existing(root)
        if not overwrite:
            msg = (
                f"dataset directory already exists at {root}"
                + (f" ({existing_summary})" if existing_summary else "")
                + ". Pass overwrite=True (or --overwrite on the CLI) to wipe it first; "
                "otherwise pick a different --repo-id / --out."
            )
            raise FileExistsError(msg)
        # Wipe before letting lerobot's `mkdir(exist_ok=False)` run, so the
        # collision is ours to report and not a stack trace from inside lerobot.
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        fps=fps,
        features=_features(config),
        root=config.root,
        robot_type="so100_follower",
        use_videos=True,
    )

    written: list[str] = []
    try:
        for episode, state, action in usable:
            if should_continue is not None and not should_continue():
                report.cancelled = True
                break
            streams = {
                role: decode_video(
                    episode.path / f"{role}.mp4", len(state), config.resolution
                )
                for role in config.cameras
            }
            for i in range(len(state)):
                frame: dict[str, Any] = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": episode.meta.task,
                }
                for role, stream in streams.items():
                    frame[f"observation.images.{role}"] = next(stream)
                # No `timestamp`: lerobot derives it as frame_index / fps, which
                # is exactly the grid this export chose the rate to match.
                dataset.add_frame(frame)
            dataset.save_episode()
            written.append(episode.meta.id)
            if progress is not None:
                progress(episode.meta.id, len(state), len(written), len(usable))
    finally:
        # Not optional, and not only on the happy path: without it the parquet
        # footers are never written and the dataset cannot be opened at all. A
        # cancelled or failed export that leaves a readable partial dataset is
        # worth far more than one that leaves an unopenable directory.
        dataset.finalize()

    if report.cancelled:
        # The report described the whole selection; it now has to describe what
        # is actually on disk, or the summary would claim episodes nobody wrote.
        kept = set(written)
        for entry in report.episodes:
            if entry.included and entry.episode_id not in kept:
                entry.included = False
                entry.reason = "cancelled before this episode was written"
        report.n_episodes = len(written)
        report.n_frames = sum(
            len(state) for episode, state, _ in usable if episode.meta.id in kept
        )

    report.dataset_path = Path(dataset.root)
    report.episode_ids = tuple(written)
    write_manifest(report, config)
    return report


def write_manifest(report: ExportReport, config: ExportConfig) -> Path:
    """Record what this export was made from, next to the dataset.

    See the module docstring: lerobot's metadata has nowhere to say which take a
    dataset episode came from, and that mapping is what makes the export
    checkable and re-runnable instead of a one-way door.
    """
    if report.dataset_path is None:
        raise ValueError("cannot write a manifest for an export that wrote nothing")
    manifest = {
        "repo_id": config.repo_id,
        "fps": report.fps,
        "action_space": report.action_space,
        "cameras": list(config.cameras),
        "resolution": list(config.resolution),
        "task": config.task,
        "n_episodes": report.n_episodes,
        "n_frames": report.n_frames,
        "cancelled": report.cancelled,
        # In dataset episode order, so `episode_ids[i]` is the take that dataset
        # episode `i` was written from. That is the whole point of the file.
        "episode_ids": list(report.episode_ids),
        "episode_root": str(config.episode_root or ""),
    }
    path = report.dataset_path / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _summarise_existing(dataset_path: Path) -> str:
    """A one-line summary of what is at `dataset_path`, for overwrite errors.

    Reads lerobot's `info.json` when present (the common case -- a real
    dataset the operator wrote earlier), and falls back to a size-only line
    when the directory is a stub or a foreign format. Returning "" would be
    worse: an operator who sees "directory exists" with no further
    information has nothing to decide against.
    """
    info_path = dataset_path / "meta" / _LEROBOT_INFO_NAME
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        else:
            n_ep = int(info.get("total_episodes", 0))
            n_fr = int(info.get("total_frames", 0))
            return f"{n_ep} episodes / {n_fr} frames"
    total = 0
    try:
        for item in dataset_path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    if total:
        return f"~{total / (1024 * 1024):.1f} MB on disk"
    return "empty directory"
    return path


def read_manifest(dataset_path: Path) -> dict[str, Any]:
    """The export manifest for a dataset, or a clear failure."""
    path = Path(dataset_path) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: this dataset was not written by so-snake's exporter, "
            "or was written before it recorded its source episodes. Re-export to "
            "make it verifiable."
        )
    return json.loads(path.read_text(encoding="utf-8"))


# Lerobot's own metadata path. Stable across lerobot versions; documented in
# `lerobot.datasets.lerobot_dataset` as the file every well-formed dataset has.
_LEROBOT_INFO_NAME = "info.json"


def _has_manifest(dataset_path: Path) -> bool:
    """Whether our `export.json` sits next to a dataset on disk."""
    return (Path(dataset_path) / MANIFEST_NAME).is_file()


def _lerobot_meta(dataset_path: Path) -> dict[str, Any]:
    """What lerobot's own `info.json` says about a dataset.

    A fallback for `dataset_meta`: fps and feature shapes are enough to make
    a replayable `Episode`, and they are exactly what lerobot itself uses when
    reading the dataset. Without this fallback a dataset that lost its
    manifest (legacy export, foreign tool) would be unreadable to us even
    though lerobot can still load it.

    Action space is inferred from the action feature's names: a `dx/dy/dz/...`
    prefix means `delta`, plain `x/y/z/...` means `absolute`. The convention
    is what this exporter writes and what `apply_action` understands; reading
    it back from the parquet keeps the two halves in agreement without
    recording it explicitly.
    """
    info_path = Path(dataset_path) / "meta" / _LEROBOT_INFO_NAME
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{info_path} is missing: this is not a LeRobotDataset directory. "
            "A dataset directory must have either so-snake's export.json "
            "or lerobot's meta/info.json (or both)."
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    action_names = list(info["features"]["action"]["names"])
    if action_names and all(action_names[i].startswith("d") for i in range(5)):
        action_space = "delta"
    else:
        action_space = "absolute"
    # Deliberately the same keys `write_manifest` produces, so that everything
    # downstream can read one shape without asking which kind it got. Two
    # near-identical dicts under one name is how a UI ends up reading a field
    # that exists on one of them: the episode picker took `n_episodes` from
    # here and got `None`, so a dataset without our manifest showed zero
    # episodes and could not be replayed at all.
    #
    # No `root`. Where a dataset lives is a property of where it was found, not
    # of its metadata, and baking a path into either would be wrong the moment
    # the directory moved. Callers already have the path -- they passed it in.
    return {
        # The repo_id is what lerobot writes into the parquet filenames; when
        # our manifest is missing, the directory name is the only handle we
        # have, and that is what an operator would have called it.
        "repo_id": Path(dataset_path).name,
        "task": None,
        "fps": int(info["fps"]),
        "action_space": action_space,
        "cameras": [
            key.removeprefix("observation.images.")
            for key in info.get("features", {})
            if key.startswith("observation.images.")
        ],
        "resolution": _lerobot_resolution(info),
        "n_episodes": int(info.get("total_episodes", 0)),
        "n_frames": int(info.get("total_frames", 0)),
        "cancelled": False,
        # `episode_ids` is the source mapping; without our manifest there is
        # no source mapping, and downstream code must skip the source-fidelity
        # branch (see `verify`).
        "episode_ids": [],
        "episode_root": "",
    }


def _lerobot_resolution(info: dict[str, Any]) -> list[int]:
    """`[height, width]` from the first image feature, or `[0, 0]`.

    Shapes are `(h, w, c)` in a LeRobotDataset's feature table, which is the
    same order `ExportConfig.resolution` uses, so this passes straight through.
    """
    for key, feature in info.get("features", {}).items():
        if not key.startswith("observation.images."):
            continue
        shape = list(feature.get("shape", ()))
        if len(shape) >= 2:
            return [int(shape[0]), int(shape[1])]
    return [0, 0]


def dataset_meta(dataset_path: Path) -> tuple[dict[str, Any], bool]:
    """The metadata a dataset carries: ours if present, lerobot's if not.

    Returns the metadata dict and `True`/`False` for "we wrote this". The
    second value lets callers distinguish "we made this and the source
    mapping is intact" from "we did not make this and the source mapping is
    unknown" -- they answer different questions.
    """
    if _has_manifest(dataset_path):
        return read_manifest(dataset_path), True
    return _lerobot_meta(dataset_path), False


@dataclass
class VerifyReport:
    """What reading the written dataset back off disk proved about it."""

    repo_id: str
    dataset_path: Path
    fps: int
    action_space: str
    n_episodes: int = 0
    n_frames: int = 0
    # How many episodes were actually compared against their source take. Fewer
    # than `n_episodes` means the error figures below cover only part of the
    # dataset, and every gap is named in `skipped`. Without this the numbers read
    # as whole-dataset claims: one comparable episode out of fifty reports the
    # same "0.00e+0" as fifty out of fifty.
    episodes_compared: int = 0
    # Worst disagreement between the rows on disk and the rows recomputed from
    # the source episodes. Non-zero by construction -- the dataset is float32
    # and the recording is float64 -- so what matters is the magnitude.
    state_max_abs_error: float = 0.0
    action_max_abs_error: float = 0.0
    # Worst error in the 5D targets reconstructed from the rows on disk, against
    # the targets the operator actually commanded. This is the replay contract.
    target_position_error_max_m: float = 0.0
    target_angle_error_max_rad: float = 0.0
    gripper_error_max_deg: float = 0.0
    timestamp_max_error_s: float = 0.0
    video_frames: dict[str, int] = field(default_factory=dict)
    # Failures: a check ran against a reference and the dataset disagreed with
    # it. `ok` is False iff this list is non-empty.
    issues: list[str] = field(default_factory=list)
    # Checks that could not run, because an input that lives outside the dataset
    # was not there (no manifest, no episode store, a source take since deleted).
    # Not failures -- nothing was learned about the dataset either way -- but the
    # operator must know, so a green `ok` is not read as "checked against source
    # takes". See `verify` for why the line falls here.
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


# float32 storage of a metre-scale quantity resolves to about 1e-8 m, and the
# reconstruction adds one subtraction and one addition on top. A millimetre of
# slack over that is still three orders of magnitude below the 9.6 mm of servo
# lag the action space is built around, so an error this small cannot be a
# contract mismatch -- and one that is larger always is.
_REPLAY_TOLERANCE_M = 1e-4
_REPLAY_TOLERANCE_RAD = 1e-4
_REPLAY_TOLERANCE_DEG = 1e-2


def verify(
    dataset_path: Path,
    store: EpisodeStore | None = None,
    *,
    so_snake_config: SoSnakeConfig | None = None,
    check_videos: bool = True,
) -> VerifyReport:
    """Read an exported dataset back off disk and prove it can be replayed.

    Not a re-run of the checks `plan` already did. Those ran on the arrays in
    memory; this opens the parquet, the manifest and the video files that were
    actually written, and asks three questions of them:

      * **do the rows survive the round trip** -- the state and action on disk
        against the same columns recomputed from the source episodes;
      * **do they still invert** -- `apply_action` over the rows on disk against
        the 5D targets the operator commanded, which is the contract a rollout
        depends on and the one thing that makes the dataset replayable rather
        than merely readable;
      * **is the time axis real** -- timestamps against `frame_index / fps`, and
        one video frame present per row per camera.

    The failures this catches are exactly the ones that live between the writer
    and the reader, and none of them raise on their own: a missing parquet
    footer, a video short by a frame, a timestamp grid built from a rate nothing
    ran at. Pass `store=None` to check only what the dataset can check about
    itself.

    ## Failure versus gap, which is the whole of `issues` versus `skipped`

    An `issue` is a check that ran and the dataset lost: the rows disagree with
    the recording, the timestamps disagree with `frame_index / fps`, the manifest
    disagrees with the parquet about how many episodes there are. Every reference
    in that list lives *inside* the dataset or is the arithmetic the format
    defines, so a disagreement is a defect of these bytes and blocks training.

    A `skipped` entry is a check that could not run because something outside the
    dataset was absent: no `export.json`, no episode store, or a source take that
    has since been deleted from the store. Nothing was learned, and nothing is
    wrong with the dataset -- deleting a take does not change a byte of it. This
    is the reason the line falls exactly here rather than at "we could not
    confirm it, so refuse it": a verdict that turns red because a *different*
    directory changed is not a statement about the dataset at all, and the same
    export would then verify green on the bench that still holds the takes and
    red on the training box that never had them. A verdict has to be a property
    of the artefact under test to be worth citing about it.

    So a dataset whose takes are gone reads PARTIAL, not FAILED -- amber, with
    every unresolvable episode named, and `episodes_compared` saying how much of
    the dataset the error figures actually cover. It is trainable; what is lost
    is the ability to audit it against the recording, or to export it again.
    A dataset without our `export.json` (foreign, legacy, or wiped by hand) is
    the same case arrived at from the other end: the parquet round-trip, the time
    axis and the video frame counts still run, because they need only the
    parquet. Either way the report is *not* marked `ok` merely because the
    round-trip passed -- "no source comparison" is a different answer from
    "source comparison passed", and the GUI shows it as a third badge.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_path = Path(dataset_path)
    manifest, ours = dataset_meta(dataset_path)
    report = VerifyReport(
        repo_id=str(manifest["repo_id"]),
        dataset_path=dataset_path,
        fps=int(manifest["fps"]),
        action_space=str(manifest["action_space"]),
    )

    dataset = LeRobotDataset(report.repo_id, root=dataset_path)
    report.n_episodes = int(dataset.num_episodes)
    report.n_frames = int(dataset.num_frames)

    if int(dataset.fps) != report.fps:
        report.issues.append(
            f"dataset fps is {dataset.fps}, manifest says {report.fps}"
        )

    episode_ids = list(manifest.get("episode_ids", ()))
    # Without our manifest there is no source mapping at all. Without a store
    # the source comparison cannot run even with a mapping. Either way, the
    # round-trip and time-axis checks below still run -- they do not depend on
    # source episodes -- and the report records what was skipped so the
    # verdict cannot be misread as "verified against source takes".
    if not ours:
        report.skipped.append(
            "export.json is missing: source-fidelity check skipped -- the "
            "parquet was checked, but no comparison against source takes ran. "
            "Re-export with this exporter to make the mapping trustable."
        )
        episode_ids = []
    elif len(episode_ids) != report.n_episodes:
        report.issues.append(
            f"manifest lists {len(episode_ids)} source episodes but the dataset has "
            f"{report.n_episodes}; the mapping from dataset episode to take is not "
            "trustworthy and nothing below could be checked against its source"
        )
        episode_ids = []
    elif store is None:
        report.skipped.append(
            "no episode store supplied: source-fidelity check skipped -- the "
            "parquet was checked, but no comparison against source takes ran. "
            "Pass an EpisodeStore to make the mapping trustable."
        )

    # Source takes the store cannot hand over, collected rather than reported one
    # by one: on a fifty-take export a whole session deleted after the fact would
    # otherwise be fifty near-identical lines in the operator's face.
    unresolved: list[str] = []

    for index in range(report.n_episodes):
        columns = dataset.get_episode_column_arrays(
            index, ["observation.state", "action", "timestamp", "frame_index"]
        )
        state = np.asarray(columns["observation.state"], dtype=float)
        action = np.asarray(columns["action"], dtype=float)
        timestamp = np.asarray(columns["timestamp"], dtype=float).reshape(-1)
        frame_index = np.asarray(columns["frame_index"], dtype=np.int64).reshape(-1)

        # lerobot derives the time axis from the row index and the dataset rate,
        # and seeks the videos by it. If it has drifted from that definition the
        # frames and the rows are no longer the same moment.
        expected_ts = frame_index / float(report.fps)
        if len(timestamp):
            report.timestamp_max_error_s = max(
                report.timestamp_max_error_s,
                float(np.abs(timestamp - expected_ts).max()),
            )

        if not episode_ids:
            continue
        source_id = episode_ids[index]
        if store is None:
            continue
        try:
            episode = store.load(source_id)
        except (FileNotFoundError, ValueError) as exc:
            # Not an issue: the take is not part of the dataset, and its absence
            # says nothing about these bytes. The exception text is kept because
            # "deleted from the store" and "on disk but unreadable" ask the
            # operator for different things.
            unresolved.append(f"episode {index} ({source_id}): {exc}")
            continue

        if len(state) != int(episode.meta.n_steps):
            report.issues.append(
                f"episode {index} ({source_id}): {len(state)} rows on disk for "
                f"{episode.meta.n_steps} recorded steps"
            )
            continue

        expected_state, expected_action, _ = build_state_action(
            episode, report.action_space, so_snake_config
        )
        report.state_max_abs_error = max(
            report.state_max_abs_error,
            float(np.abs(state - np.asarray(expected_state, dtype=float)).max()),
        )
        report.action_max_abs_error = max(
            report.action_max_abs_error,
            float(np.abs(action - np.asarray(expected_action, dtype=float)).max()),
        )

        # The replay contract, run over the rows as stored.
        targets, gripper = replay_targets_from_state_action(
            state, action, report.action_space
        )
        expected_targets = np.asarray(episode.task_target, dtype=float)
        diff = np.asarray(targets, dtype=float) - expected_targets
        for dim in _ANGULAR_DIMS:
            diff[:, dim] = wrap_to_pi(diff[:, dim])
        report.target_position_error_max_m = max(
            report.target_position_error_max_m,
            float(np.linalg.norm(diff[:, :3], axis=1).max()),
        )
        report.target_angle_error_max_rad = max(
            report.target_angle_error_max_rad, float(np.abs(diff[:, 3:5]).max())
        )
        report.gripper_error_max_deg = max(
            report.gripper_error_max_deg,
            float(
                np.abs(
                    np.asarray(gripper, dtype=float)
                    - np.asarray(episode.gripper_cmd_deg, dtype=float)
                ).max()
            ),
        )
        report.episodes_compared += 1

    if unresolved:
        report.skipped.append(_unresolved_note(unresolved, report))

    if check_videos:
        report.video_frames = _verify_videos(dataset_path, manifest, report)

    if report.timestamp_max_error_s > 1.0 / (2.0 * report.fps):
        report.issues.append(
            f"timestamps drift from frame_index/fps by up to "
            f"{report.timestamp_max_error_s * 1000:.1f} ms, over half a frame; "
            "lerobot seeks the videos by this axis, so rows and frames no longer "
            "refer to the same moment"
        )
    if report.target_position_error_max_m > _REPLAY_TOLERANCE_M:
        report.issues.append(
            f"replaying the stored rows misses the commanded targets by up to "
            f"{report.target_position_error_max_m * 1000:.3f} mm; the action space "
            "on disk does not invert to what was recorded"
        )
    if report.target_angle_error_max_rad > _REPLAY_TOLERANCE_RAD:
        report.issues.append(
            f"replaying the stored rows misses the commanded pitch/roll by up to "
            f"{np.degrees(report.target_angle_error_max_rad):.4f} deg"
        )
    if report.gripper_error_max_deg > _REPLAY_TOLERANCE_DEG:
        report.issues.append(
            f"gripper angle differs by up to {report.gripper_error_max_deg:.3f} deg"
        )
    return report


# How many unresolvable takes to name before summarising. Enough that a couple of
# deleted takes are identified outright -- which is what the operator needs to
# decide whether they care -- without turning a wiped session into a wall.
_UNRESOLVED_SHOWN = 5


def _unresolved_note(unresolved: list[str], report: VerifyReport) -> str:
    """The `skipped` entry for source takes the store could not hand over.

    Says three things, because each drives a different decision: how much of the
    dataset went unchecked (is this dataset still worth citing), which takes are
    gone (can they be recovered), and what the numbers in the report now cover
    (are the errors below a whole-dataset claim -- they are not).
    """
    shown = unresolved[:_UNRESOLVED_SHOWN]
    rest = len(unresolved) - len(shown)
    listed = "; ".join(shown) + (f"; and {rest} more" if rest else "")
    return (
        f"{len(unresolved)}/{report.n_episodes} episodes could not be compared against "
        "their source take: the take is no longer readable in the store, so nothing "
        "checked those rows against what was recorded. The dataset itself is "
        "unaffected -- it is still loadable and trainable -- but it can no longer be "
        "audited against the recording or re-exported, and the error figures in this "
        f"report cover only the {report.episodes_compared} episode(s) that were "
        f"compared. Unresolved: {listed}"
    )


def _verify_videos(
    dataset_path: Path, manifest: dict[str, Any], report: VerifyReport
) -> dict[str, int]:
    """One decodable frame per row, per camera. Counted, not trusted.

    The recorder's contract is that video frame *i* is row *i*; a video one
    frame short slides every subsequent frame against its state by one step,
    which is invisible in training and shows up as a policy that acts early.
    """
    counts: dict[str, int] = {}
    for role in manifest.get("cameras", ()):
        key = f"observation.images.{role}"
        total = 0
        directory = dataset_path / "videos" / key
        if not directory.is_dir():
            report.issues.append(f"{role}: no videos written at {directory}")
            counts[role] = 0
            continue
        for path in sorted(directory.rglob("*.mp4")):
            try:
                total += _count_video_frames(path)
            except Exception as exc:  # noqa: BLE001 - the reason is the payload
                report.issues.append(f"{path.name} ({role}) could not be decoded: {exc}")
        counts[role] = total
        if total != report.n_frames:
            report.issues.append(
                f"{role}: {total} video frames for {report.n_frames} rows; the "
                "recorder's frame-i-is-row-i contract does not hold on disk"
            )
    return counts


def _count_video_frames(path: Path) -> int:
    """Decoded frame count. Packets are counted where the container agrees.

    Container metadata is not trusted here for the same reason the export
    re-encodes at a measured rate: this repository has already been bitten by an
    mp4 header that disagreed with its own contents.
    """
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        return sum(1 for _ in container.decode(stream))


def _action_stats(actions: np.ndarray) -> dict[str, Any]:
    """Spread of the manifold columns, and separately what the gripper does.

    The zero share on the manifold columns is the number to look at before
    training a delta policy: a column that is mostly zero trains a policy that
    mostly outputs zero, and under an L1 objective that is a genuine minimum
    rather than a failure to converge.

    The gripper gets different numbers because it is a different kind of
    quantity -- an absolute angle, not a step. Percentiles of it describe the
    duty cycle of an open jaw and say nothing about whether the takes contain a
    grasp at all, so what is reported instead is the share of steps commanded
    closed and how many times it crossed. A selection with no crossings has no
    grasp in it, however much motion it contains.
    """
    out: dict[str, Any] = {}
    for i, name in enumerate(TASK_DIM_NAMES):
        column = actions[:, i]
        magnitude = np.abs(column)
        out[name] = {
            "p50": float(np.percentile(magnitude, 50)),
            "p95": float(np.percentile(magnitude, 95)),
            "max": float(magnitude.max()),
            "std": float(column.std()),
            "zero_frac": float((magnitude < 1e-9).mean()),
        }
    out["all_zero_frac"] = float(
        (np.abs(actions[:, : len(TASK_DIM_NAMES)]).max(axis=1) < 1e-9).mean()
    )

    gripper = actions[:, len(TASK_DIM_NAMES)]
    midpoint = 0.5 * (float(gripper.min()) + float(gripper.max()))
    closed = gripper < midpoint
    out["gripper"] = {
        "min": float(gripper.min()),
        "max": float(gripper.max()),
        "closed_frac": float(closed.mean()),
        "transitions": int(np.count_nonzero(np.diff(closed.astype(np.int8)))),
    }
    return out


def _replay_check(
    usable: list[tuple[Episode, np.ndarray, np.ndarray]], action_space: str
) -> dict[str, Any]:
    """Can exported state/action reconstruct the demonstrations' targets?"""
    max_pos_m = 0.0
    max_angle_rad = 0.0
    max_gripper_deg = 0.0
    n_frames = 0
    for episode, state, action in usable:
        targets, gripper = replay_targets_from_state_action(state, action, action_space)
        expected = np.asarray(episode.task_target, dtype=float)
        diff = targets - expected
        for dim in _ANGULAR_DIMS:
            diff[:, dim] = wrap_to_pi(diff[:, dim])
        if len(diff):
            max_pos_m = max(max_pos_m, float(np.linalg.norm(diff[:, :3], axis=1).max()))
            max_angle_rad = max(max_angle_rad, float(np.abs(diff[:, 3:5]).max()))
            max_gripper_deg = max(
                max_gripper_deg,
                float(np.abs(gripper - np.asarray(episode.gripper_cmd_deg, dtype=float)).max()),
            )
            n_frames += len(diff)
    return {
        "frames": n_frames,
        "target_position_error_max_m": max_pos_m,
        "target_angle_error_max_rad": max_angle_rad,
        "gripper_error_max_deg": max_gripper_deg,
        "ok": max_pos_m < 1e-5 and max_angle_rad < 1e-5 and max_gripper_deg < 1e-4,
    }


def format_report(report: ExportReport, config: ExportConfig) -> str:
    """The export as text: what went in, what did not, and what it looks like."""
    lines: list[str] = []
    height, width = config.resolution
    lines.append(
        f"exported {report.n_episodes} episodes / {report.n_frames} frames "
        f"at {report.fps} Hz, action space {report.action_space!r}, "
        f"{height}x{width}, cameras {', '.join(config.cameras)}"
    )
    if report.dataset_path:
        lines.append(f"  -> {report.dataset_path}")

    included = [e for e in report.episodes if e.included]
    if included:
        drift = max(abs(e.measured_fps - report.fps) for e in included)
        lines.append(
            f"  rate: takes ran {min(e.measured_fps for e in included):.2f}"
            f"-{max(e.measured_fps for e in included):.2f} Hz, "
            f"worst deviation from {report.fps} Hz is {drift / report.fps * 100:.1f}%"
        )

        # The dataset rate is a property of the *recording*, and the only thing
        # that explains one the operator did not expect is the loop having
        # missed its configured rate at the time. Saying it here is what stops
        # the next person re-running the export hoping for a different number:
        # there isn't one, short of recording the take again.
        behind = [
            e for e in included
            if e.configured_hz > 0 and e.measured_fps < e.configured_hz * 0.95
        ]
        if behind:
            configured = sorted({round(e.configured_hz) for e in behind})
            lines.append(
                f"    ^ {len(behind)}/{len(included)} were recorded against a configured "
                f"{', '.join(f'{c} Hz' for c in configured)} and did not hold it. That is "
                "baked into those takes -- the export reports what the arm actually did, "
                "so re-exporting cannot raise it. Re-record them to get the configured rate."
            )
    assumed = [e.episode_id for e in included if not e.gripper_measured]
    if assumed:
        lines.append(
            f"  gripper state is the COMMANDED angle for {len(assumed)}/{len(included)} "
            "episodes (recorded before format v2 stored the measured one)"
        )

    if report.skipped:
        lines.append(f"  skipped {len(report.skipped)}:")
        for entry in report.skipped:
            lines.append(f"    {entry.episode_id}: {entry.reason}")

        # A take rejected purely on rate is a different kind of rejection from a
        # missing camera, and after a change to the loop's timing it is usually
        # not one bad take -- it is a whole second batch recorded at the new
        # rate. Saying so beats leaving the operator to notice that all the
        # rejects happen to share a date.
        off_rate = [e for e in report.skipped if "dataset is" in e.reason]
        if off_rate:
            rates = sorted({round(e.measured_fps) for e in off_rate})
            lines.append(
                f"    ^ {len(off_rate)} of these were rejected only for their rate "
                f"({', '.join(f'{r} Hz' for r in rates)} against this dataset's "
                f"{report.fps} Hz). Two rates cannot share one time grid, so export "
                "each batch separately rather than trying to widen fps_tolerance."
            )

    if report.action_stats:
        lines.append("  action |magnitude| per manifold dimension:")
        lines.append(f"    {'dim':8}{'p50':>10}{'p95':>10}{'max':>10}{'zero%':>8}")
        for name in TASK_DIM_NAMES:
            s = report.action_stats[name]
            lines.append(
                f"    {name:8}{s['p50']:10.5f}{s['p95']:10.5f}{s['max']:10.5f}"
                f"{s['zero_frac'] * 100:8.1f}"
            )
        all_zero = report.action_stats["all_zero_frac"]
        lines.append(f"    {all_zero * 100:.1f}% of steps ask for no manifold motion at all")
        if all_zero > 0.2 and report.action_space == "delta":
            lines.append(
                "    ^ high. The operator moves in bursts and the clutch gates the rest; "
                "expect the policy to under-move, and compare against --action-space absolute."
            )

        grip = report.action_stats["gripper"]
        lines.append(
            f"  gripper: {grip['min']:.1f}-{grip['max']:.1f} deg, "
            f"closed {grip['closed_frac'] * 100:.0f}% of steps, "
            f"{grip['transitions']} open/close crossings"
        )
        if grip["transitions"] == 0:
            lines.append(
                "    ^ the jaw never crosses. There is no grasp in this selection, so "
                "no policy trained on it can learn one."
            )
    if report.replay_check:
        check = report.replay_check
        lines.append(
            "  replay check: "
            f"{check['frames']} rows reconstruct targets with max "
            f"{check['target_position_error_max_m'] * 1000.0:.4f} mm / "
            f"{np.degrees(check['target_angle_error_max_rad']):.4f} deg / "
            f"{check['gripper_error_max_deg']:.4f} deg gripper"
        )
    return "\n".join(lines)


def episode_from_dataset(
    dataset_path: Path,
    episode_index: int = 0,
    *,
    so_snake_config: SoSnakeConfig | None = None,
) -> Episode:
    """Rebuild a replayable `Episode` from one episode of an exported dataset.

    This is what makes an export replayable rather than merely loadable: the
    rows on disk go back through `apply_action` into the 5D target stream the
    operator commanded, and the result is handed to the same `EpisodeReplayer`
    that plays a recorded take -- with the same rate-limited approach, the same
    deg/s clamp, the same joint limits and the same mesh clearance check. None
    of that safety logic is re-implemented here, which is the point; a second
    copy of it is exactly the thing that would drift out of step.

    Only `task` mode is meaningful on the result. The dataset deliberately
    carries no joint stream -- the policy is trained in task space -- so the
    joints here are solved by *today's* IK from the targets, which is precisely
    what task-mode replay does anyway. `joint` mode would be replaying this
    function's own arithmetic and would prove nothing.

    A dataset that lost its `export.json` (foreign, legacy, or wiped by hand)
    is still replayable, because everything replay needs -- fps, action space,
    the parquet itself -- lives in lerobot's own `meta/info.json`. The
    source-take mapping is the only thing the manifest adds, and replay has no
    use for it; the rebuilt `Episode` simply has no `source_id` in its notes.
    """
    from ..m3_safety.ik5d import TaskIK5D
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_path = Path(dataset_path)
    meta, ours = dataset_meta(dataset_path)
    fps = int(meta["fps"])
    action_space = str(meta["action_space"])

    dataset = LeRobotDataset(str(meta["repo_id"]), root=dataset_path)
    if not 0 <= episode_index < dataset.num_episodes:
        raise ValueError(
            f"episode {episode_index} is out of range; the dataset has "
            f"{dataset.num_episodes}"
        )
    columns = dataset.get_episode_column_arrays(
        episode_index, ["observation.state", "action"]
    )
    state = np.asarray(columns["observation.state"], dtype=float)
    action = np.asarray(columns["action"], dtype=float)
    targets, gripper = replay_targets_from_state_action(state, action, action_space)
    targets = np.asarray(targets, dtype=float)

    config = so_snake_config or SoSnakeConfig()
    ik = TaskIK5D(arm=config.arm, teleop=config.teleop, ik=config.ik)

    # Seeded from the previous solution, exactly as the teleop loop does: a
    # per-frame solve seeded from scratch is free to hop between elbow branches
    # between one row and the next, and the replayer would then be asked to
    # approach a first frame that has nothing to do with the second.
    joints = np.zeros((len(targets), len(config.arm.joint_names)), dtype=float)
    seed = np.asarray(ik.seed_for(SO100TaskPose.from_array(targets[0])), dtype=float)
    for i, target in enumerate(targets):
        result = ik.solve(seed, SO100TaskPose.from_array(target), rate_reference_deg=seed)
        joints[i] = result.joints_deg
        seed = result.joints_deg

    # Source mapping is the manifest's one contribution to replay: when ours,
    # we name the take this came from. Without ours, we still build the
    # episode -- the parquet is the source of truth for replay.
    source_ids = list(meta.get("episode_ids", ()))
    source_id = source_ids[episode_index] if episode_index < len(source_ids) else ""
    notes = f"rebuilt from the exported dataset at {dataset_path}"
    if source_id:
        notes += f"; originally recorded as {source_id}"
    elif not ours:
        notes += "; no source mapping (so-snake's export.json is missing)"
    meta_obj = EpisodeMeta(
        id=f"{dataset_path.name}#{episode_index}",
        name=f"{meta['repo_id']} episode {episode_index}",
        task=str(meta.get("task") or ""),
        notes=notes,
        n_steps=len(targets),
        # The dataset's grid is the take's measured rate, so this is the rate it
        # was recorded at -- which is what the replayer paces playback by.
        duration_s=len(targets) / float(fps),
        control_hz=float(fps),
        joint_names=(*config.arm.joint_names, "gripper"),
    )
    frames = {
        "action.task.target": targets,
        "action.task.gripper_deg": np.asarray(gripper, dtype=float),
        "action.joint.commanded_deg": joints,
        "observation.state.task_pose": np.asarray(state[:, :5], dtype=float),
    }
    return Episode(meta=meta_obj, frames=frames, path=dataset_path)


def format_verify(report: VerifyReport) -> str:
    """The verification as text: what was read back, and what it proved."""
    lines = [
        f"verified {report.repo_id} at {report.dataset_path}",
        f"  {report.n_episodes} episodes / {report.n_frames} frames at {report.fps} Hz, "
        f"action space {report.action_space!r}",
    ]
    if report.video_frames:
        lines.append(
            "  video frames: "
            + ", ".join(f"{role} {n}" for role, n in sorted(report.video_frames.items()))
            + f" (rows {report.n_frames})"
        )
    # Coverage qualifies every error figure that follows, so it is printed with
    # them rather than left to the PARTIAL note at the bottom: "match to 0" over
    # no compared episodes is not the same reading as over all of them.
    lines.append(
        f"  rows on disk match a fresh conversion to "
        f"{report.state_max_abs_error:.3g} (state) / "
        f"{report.action_max_abs_error:.3g} (action), over "
        f"{report.episodes_compared}/{report.n_episodes} episode(s) compared "
        "against their source take"
    )
    # Micrometres and microdegrees, because the honest answer here is float32
    # rounding and printing it in millimetres rounds it to a zero that looks
    # like the check never ran.
    lines.append(
        "  replay from disk reconstructs the commanded targets to "
        f"{report.target_position_error_max_m * 1e6:.3g} um / "
        f"{np.degrees(report.target_angle_error_max_rad) * 1e6:.3g} udeg / "
        f"{report.gripper_error_max_deg * 1e6:.3g} udeg gripper"
    )
    lines.append(
        "  timestamps sit on frame_index/fps to "
        f"{report.timestamp_max_error_s * 1e6:.3g} us"
    )
    if report.issues:
        lines.append(f"  NOT REPLAYABLE -- {len(report.issues)} problem(s):")
        lines.extend(f"    {issue}" for issue in report.issues)
    elif report.skipped:
        # Round-trip passes; the source-fidelity branch is what was skipped.
        # Saying only "OK" would make a no-source-mapping run look like a
        # checked one -- that is exactly the silent failure this distinction
        # exists to prevent.
        lines.append(
            f"  PARTIAL -- round-trip ok, {len(report.skipped)} check(s) skipped:"
        )
        lines.extend(f"    {note}" for note in report.skipped)
        lines.append(
            "  to make this a full check, re-export with this exporter so the "
            "source-take mapping is recorded -- which needs the source takes to "
            "still be in the store."
        )
    else:
        lines.append("  OK -- this dataset replays back to what was recorded")
    return "\n".join(lines)


def apply_action(
    pose: np.ndarray, action: np.ndarray, action_space: str
) -> tuple[np.ndarray, float]:
    """Invert the export: policy output plus current pose -> `(target5, gripper)`.

    The rollout runner's half of the action-space contract. Keeping it next to
    `build_state_action` is the point -- an anchor that disagrees between
    training and rollout is silent, and shows up only as an arm that creeps.
    """
    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"expected a {ACTION_DIM}-vector action, got {action.shape[0]}")
    task, gripper = action[: len(TASK_DIM_NAMES)], float(action[-1])
    if action_space == "absolute":
        target = task.copy()
    else:
        target = np.asarray(pose, dtype=float).reshape(-1) + task
    for dim in _ANGULAR_DIMS:
        target[dim] = float(wrap_to_pi(target[dim]))
    return target, gripper
