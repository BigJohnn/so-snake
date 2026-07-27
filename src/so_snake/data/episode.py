"""The on-disk episode format, and the column contract that defines it.

One episode is one directory:

    data/episodes/<episode_id>/
        meta.json     what the run was, how it was configured, how it went
        frames.npz    one row per control step, columns named below

## Why npz and not parquet

`numpy` is this repository's only base dependency, and the offline gates have to
keep running on a laptop with nothing else installed. `np.savez_compressed`
gives columnar, typed, self-describing storage for the cost of an import that is
already there. Recording is the one part of the pipeline that must never fail
for an environment reason -- a dropped episode cannot be re-taken once the
operator and the arm have moved on.

`LeRobotDataset` remains the training-time format. Exporting these episodes into
it is a conversion, and belongs on the training machine where lerobot is
installed, not in the recording path. That converter is still to be written.

## The column contract

The names are the dataset layout `TeleopLoop` already documents, made literal:

    action.raw.*          exactly what the device reported
    action.task.*         the policy's training target
    action.joint.*        what was sent to the servos
    observation.state.*   what came back
    diagnostics.*         why the chain did what it did

Nothing here is derived from anything else in the file, deliberately. Storing
`action.task.target` *and* `observation.state.task_pose` *and*
`action.joint.commanded_deg` is redundant only if the IK never changes; the
point of recording all three is that when it does, the old episodes can be
replayed through the new solver and the two compared. Recomputing a column costs
kilobytes; not having it costs the recording session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import REPO_ROOT
from ..teleop.loop import StepRecord

DEFAULT_EPISODE_ROOT = REPO_ROOT / "data" / "episodes"

META_NAME = "meta.json"
FRAMES_NAME = "frames.npz"

# Bumped whenever a column changes meaning. Readers refuse anything newer than
# they know; anything older is either migrated or reported, never guessed at.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Column:
    """One column of `frames.npz`: its name, where it comes from, and its type."""

    name: str
    extract: Callable[[StepRecord], Any]
    dtype: str
    width: int = 1  # 1 for a scalar column, N for an (n_steps, N) column


def _raw(key: str) -> Callable[[StepRecord], Any]:
    return lambda record: record.raw[key]


COLUMNS: tuple[Column, ...] = (
    Column("index", lambda r: r.index, "i8"),
    Column("t", lambda r: r.t, "f8"),
    # -- action.raw: the device frame, robot-agnostic ------------------------
    Column("action.raw.t", _raw("action.raw.t"), "f8"),
    Column("action.raw.sticks", _raw("action.raw.sticks"), "f8", 4),
    Column("action.raw.imu_quaternion", _raw("action.raw.imu_quaternion"), "f8", 4),
    Column("action.raw.clutch", _raw("action.raw.clutch"), "?"),
    Column("action.raw.gripper", _raw("action.raw.gripper"), "f8"),
    # -- action.task: the 5D target the policy is trained to emit ------------
    Column("action.task.target", lambda r: r.task_target, "f8", 5),
    Column("action.task.delta", lambda r: r.task_delta, "f8", 5),
    Column("action.task.gripper_deg", lambda r: r.gripper_cmd_deg, "f8"),
    # -- action.joint: what crossed the M4 boundary --------------------------
    Column("action.joint.commanded_deg", lambda r: r.commanded_joints_deg, "f8", 5),
    # -- observation.state: what came back -----------------------------------
    Column("observation.state.joints_deg", lambda r: r.measured_joints_deg, "f8", 5),
    Column("observation.state.task_pose", lambda r: r.achieved_task_pose, "f8", 5),
    Column("observation.state.position", lambda r: r.achieved_position, "f8", 3),
    Column("observation.state.quaternion", lambda r: r.achieved_quaternion, "f8", 4),
    # -- diagnostics: why the chain did what it did --------------------------
    Column("diagnostics.ik_position_error_m", lambda r: r.ik_position_error_m, "f8"),
    Column("diagnostics.ik_pitch_error_rad", lambda r: r.ik_pitch_error_rad, "f8"),
    Column("diagnostics.ik_roll_error_rad", lambda r: r.ik_roll_error_rad, "f8"),
    Column("diagnostics.projected_pitch_delta", lambda r: r.projected_pitch_delta, "f8"),
    Column("diagnostics.projected_roll_delta", lambda r: r.projected_roll_delta, "f8"),
    Column("diagnostics.rejected_rotation_norm", lambda r: r.rejected_rotation_norm, "f8"),
    Column("diagnostics.yaw_residual_rad", lambda r: r.yaw_residual_rad, "f8"),
    Column("diagnostics.orientation_saturated", lambda r: r.orientation_saturated, "?"),
    Column("diagnostics.workspace_clamped", lambda r: r.workspace_clamped, "?"),
    Column("diagnostics.atlas_pitch_clamped", lambda r: r.atlas_pitch_clamped, "?"),
    Column("diagnostics.atlas_roll_infeasible", lambda r: r.atlas_roll_infeasible, "?"),
    Column("diagnostics.joint_limit_clamped", lambda r: r.joint_limit_clamped, "?"),
    Column("diagnostics.joint_rate_clamped", lambda r: r.joint_rate_clamped, "?"),
    Column("diagnostics.command_safety_held", lambda r: r.command_safety_held, "?"),
    Column("diagnostics.command_safety_reason", lambda r: r.command_safety_reason, "U64"),
    # NaN where the backend has no geometry to probe -- the mock and the real
    # arm both report nothing here, only MuJoCo does.
    Column(
        "diagnostics.robot_mesh_min_z_m",
        lambda r: np.nan if r.robot_mesh_min_z_m is None else r.robot_mesh_min_z_m,
        "f8",
    ),
    Column("diagnostics.robot_mesh_min_body", lambda r: r.robot_mesh_min_body, "U32"),
    Column("diagnostics.ik_converged", lambda r: r.ik_converged, "?"),
    Column("diagnostics.ik_solver_converged", lambda r: r.ik_solver_converged, "?"),
    Column("diagnostics.ik_reseeded", lambda r: r.ik_reseeded, "?"),
    Column("diagnostics.ik_iterations", lambda r: r.ik_iterations, "i8"),
    Column("diagnostics.ik_min_singular_value", lambda r: r.ik_min_singular_value, "f8"),
    Column("diagnostics.clutch_engaged", lambda r: r.clutch_engaged, "?"),
    Column("diagnostics.loop_dt_s", lambda r: r.loop_dt_s, "f8"),
)

COLUMN_NAMES: tuple[str, ...] = tuple(column.name for column in COLUMNS)


def encode_frames(records: list[StepRecord]) -> dict[str, np.ndarray]:
    """Turn step records into the column arrays that go into `frames.npz`."""
    frames: dict[str, np.ndarray] = {}
    for column in COLUMNS:
        values = [column.extract(record) for record in records]
        shape = (len(records),) if column.width == 1 else (len(records), column.width)
        frames[column.name] = np.array(values, dtype=column.dtype).reshape(shape)
    return frames


@dataclass
class EpisodeMeta:
    """Everything about a run that is not a per-step number.

    `config` is a snapshot rather than a reference: the workspace box, the rate
    caps and the home pose all change during tuning, and an episode recorded
    under the old ones has to stay interpretable. Reading the current config at
    replay time would silently reinterpret old data.
    """

    id: str
    name: str = ""
    task: str = ""
    notes: str = ""
    created_at: str = ""
    format_version: int = FORMAT_VERSION

    backend: str = ""  # mock | mujoco | real
    source: str = ""  # scripted | pro
    simulated: bool = True

    n_steps: int = 0
    duration_s: float = 0.0
    control_hz: float = 0.0
    joint_names: list[str] = field(default_factory=list)

    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)

    # Set when a recording ends for a reason other than the operator asking it
    # to -- a full disk, a step cap, a backend that dropped out. An episode with
    # this set is still readable; it is just not necessarily a complete demo.
    aborted_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task": self.task,
            "notes": self.notes,
            "created_at": self.created_at,
            "format_version": self.format_version,
            "backend": self.backend,
            "source": self.source,
            "simulated": self.simulated,
            "n_steps": self.n_steps,
            "duration_s": self.duration_s,
            "control_hz": self.control_hz,
            "joint_names": list(self.joint_names),
            "config": self.config,
            "summary": self.summary,
            "aborted_reason": self.aborted_reason,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> EpisodeMeta:
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001 - dataclass API
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class Episode:
    """A recorded episode, loaded from disk."""

    meta: EpisodeMeta
    frames: dict[str, np.ndarray]
    path: Path

    def __len__(self) -> int:
        return int(self.meta.n_steps)

    def column(self, name: str) -> np.ndarray:
        if name not in self.frames:
            raise KeyError(f"episode {self.meta.id} has no column {name!r}")
        return self.frames[name]

    @property
    def commanded_joints_deg(self) -> np.ndarray:
        """`(n, 5)` arm joints as executed, without the gripper."""
        return self.column("action.joint.commanded_deg")

    @property
    def gripper_cmd_deg(self) -> np.ndarray:
        return self.column("action.task.gripper_deg")

    @property
    def task_target(self) -> np.ndarray:
        """`(n, 5)` the `(x, y, z, pitch, roll)` targets, post-clamp and post-atlas."""
        return self.column("action.task.target")

    def full_command_deg(self) -> np.ndarray:
        """`(n, 6)` arm joints with the gripper appended, in backend order."""
        return np.concatenate(
            [self.commanded_joints_deg, self.gripper_cmd_deg.reshape(-1, 1)], axis=1
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_episode(root: Path, meta: EpisodeMeta, frames: dict[str, np.ndarray]) -> Path:
    """Write one episode directory. Frames land before meta, always.

    A directory with `frames.npz` but no `meta.json` is an incomplete write and
    the store skips it. Doing it the other way round would make a half-written
    episode look complete.
    """
    path = root / meta.id
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / FRAMES_NAME, **frames)
    (path / META_NAME).write_text(
        json.dumps(meta.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def read_meta(path: Path) -> EpisodeMeta:
    payload = json.loads((path / META_NAME).read_text(encoding="utf-8"))
    version = int(payload.get("format_version", 0))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"episode {path.name} is format version {version}, this build reads up to "
            f"{FORMAT_VERSION}"
        )
    return EpisodeMeta.from_json(payload)


def read_episode(path: Path) -> Episode:
    meta = read_meta(path)
    with np.load(path / FRAMES_NAME, allow_pickle=False) as data:
        frames = {name: data[name] for name in data.files}
    return Episode(meta=meta, frames=frames, path=path)
