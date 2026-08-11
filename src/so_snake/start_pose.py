"""The pose the arm goes back to, and how it gets recorded.

Homing needs a target, and the useful one is not a constant. `TeleopConfig.
home_joints_deg` is a safe *default* -- inside the joint limits, arm folded
somewhere harmless -- but the pose a session should return to is a property of
the workcell: where the bin is, how the camera is aimed, where the operator
wants every take to start from. That pose is found by flying the arm there once
and looking at it, which is exactly what this module makes recordable:
`assets/so100_start_pose.json` is written from the arm's own joints, and homing
reads it back.

Two properties this file has to have, because homing commands it blind:

**It is validated on the way in, not on the way out.** A pose outside the joint
limits saved now is a homing move that drives into a limit clamp later, with
nobody watching -- the operator who recorded it has long since moved on. So
`save_start_pose` refuses, and `load_start_pose` refuses again on read, because
the file is editable by hand and by whoever pulls the repo next.

**It says which frame it is in.** URDF degrees, the control frame -- the same
one `SOFollowerBackend(joint_map=...)` accepts and the loop's kinematics work
in. lerobot's own calibration frame is a different set of numbers for the same
physical pose, and a file that did not say which one it held would eventually be
read as the other.

The workspace box is recorded but not enforced. The bench's own recorded start
pose sits outside it (the box is the *teleoperation* clamp, and the arm can
legitimately start folded outside it and fly in), so refusing would reject a
pose the operator deliberately chose. It is reported instead.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import ARM_JOINTS, GRIPPER_JOINT, REPO_ROOT, ArmConfig, TaskLimits

DEFAULT_START_POSE_PATH = REPO_ROOT / "assets" / "so100_start_pose.json"

# The order every array in this module is in: the backend's `joint_names`.
JOINT_ORDER: tuple[str, ...] = (*ARM_JOINTS, GRIPPER_JOINT)

CONVENTION = "URDF degrees (control frame); command through SOFollowerBackend(joint_map)"


class StartPoseError(ValueError):
    """The pose on disk, or the one being saved, cannot be homed to."""


def _limits(arm: ArmConfig) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([arm.joint_limits_deg[j][0] for j in JOINT_ORDER], dtype=float)
    hi = np.array([arm.joint_limits_deg[j][1] for j in JOINT_ORDER], dtype=float)
    return lo, hi


def check_joint_limits(joints_deg: Sequence[float], arm: ArmConfig | None = None) -> list[str]:
    """Joints outside their URDF limits, named. Empty means the pose is homeable."""
    arm = arm or ArmConfig()
    values = np.asarray(joints_deg, float)
    if values.shape != (len(JOINT_ORDER),):
        raise StartPoseError(f"expected {len(JOINT_ORDER)} joints in order {JOINT_ORDER}")
    lo, hi = _limits(arm)
    return [
        f"{name} {value:.1f} outside [{low:.1f}, {high:.1f}]"
        for name, value, low, high in zip(JOINT_ORDER, values, lo, hi)
        if not (low <= value <= high)
    ]


def save_start_pose(
    joints_deg: Sequence[float],
    *,
    path: Path = DEFAULT_START_POSE_PATH,
    arm: ArmConfig | None = None,
    limits: TaskLimits | None = None,
    task_pose: Sequence[float] | None = None,
    source: str = "",
) -> dict[str, Any]:
    """Write `joints_deg` (URDF degrees, `JOINT_ORDER`) as the start pose.

    Returns the written document, so a caller can show the operator exactly what
    landed on disk rather than re-reading it. `task_pose` is the 5D readout when
    the caller has one -- it is recorded for the operator's benefit (is this
    pose inside the teleoperation box?) and is never read back as a target;
    homing is joint-space precisely so that it needs no IK.
    """
    arm = arm or ArmConfig()
    values = np.asarray(joints_deg, float)
    out_of_range = check_joint_limits(values, arm)
    if out_of_range:
        raise StartPoseError(
            "refusing to record a start pose outside the joint limits: "
            + "; ".join(out_of_range)
        )

    document: dict[str, Any] = {
        "convention": CONVENTION,
        "joints_urdf_deg": {name: round(float(v), 3) for name, v in zip(JOINT_ORDER, values)},
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "recorded_from": source,
        "recorded_from_map": "assets/so100_joint_map.json",
        "in_joint_limits": True,
    }
    if task_pose is not None:
        pose = np.asarray(task_pose, float)
        box = limits or TaskLimits()
        inside = bool(
            np.all(pose[:3] >= np.asarray(box.pos_min_m)) and np.all(pose[:3] <= np.asarray(box.pos_max_m))
        )
        document["task_pose_xyz_m"] = [round(float(v), 4) for v in pose[:3]]
        # Recorded, not enforced: see the module docstring. A start pose outside
        # the teleoperation box is a normal thing to want.
        document["in_workspace_box"] = inside

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return document


def read_start_pose(path: Path = DEFAULT_START_POSE_PATH) -> dict[str, Any] | None:
    """The raw document, or None when there is no start pose on disk."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StartPoseError(f"start pose at {path} is unreadable: {exc}") from exc
    if not isinstance(document, dict) or "joints_urdf_deg" not in document:
        raise StartPoseError(f"start pose at {path} has no joints_urdf_deg")
    return document


def load_start_pose(
    path: Path = DEFAULT_START_POSE_PATH,
    arm: ArmConfig | None = None,
) -> np.ndarray | None:
    """The start pose as a `JOINT_ORDER` array, or None if there is not one.

    Raises rather than returning a partial pose when the file names joints this
    arm does not have, or holds one outside the limits: homing commands this
    blind, and "mostly the right pose" is a move into a limit clamp.
    """
    document = read_start_pose(path)
    if document is None:
        return None
    joints = document["joints_urdf_deg"]
    missing = [name for name in JOINT_ORDER if name not in joints]
    if missing:
        raise StartPoseError(f"start pose at {path} is missing joints: {', '.join(missing)}")
    values = np.array([float(joints[name]) for name in JOINT_ORDER])
    out_of_range = check_joint_limits(values, arm)
    if out_of_range:
        raise StartPoseError(
            f"start pose at {path} is outside the joint limits: " + "; ".join(out_of_range)
        )
    return values


def describe_start_pose(path: Path = DEFAULT_START_POSE_PATH) -> dict[str, Any]:
    """What the UI shows: where homing will go, and why.

    Never raises. A start pose that cannot be used is a thing the operator has
    to be *told* about -- silently homing to the configured default while the
    screen shows a recorded pose is the failure this exists to prevent -- so the
    reason travels with it and the caller can show it in red.
    """
    path = Path(path)
    try:
        document = read_start_pose(path)
    except StartPoseError as exc:
        return {"source": "config", "path": str(path), "error": str(exc), "joints_deg": {}}
    if document is None:
        return {"source": "config", "path": str(path), "error": "", "joints_deg": {}}
    try:
        load_start_pose(path)
    except StartPoseError as exc:
        return {
            "source": "config",
            "path": str(path),
            "error": str(exc),
            "joints_deg": dict(document.get("joints_urdf_deg", {})),
        }
    return {
        "source": "file",
        "path": str(path),
        "error": "",
        "joints_deg": {k: float(v) for k, v in document["joints_urdf_deg"].items()},
        "recorded_at": str(document.get("recorded_at", "")),
        "recorded_from": str(document.get("recorded_from", "")),
        "task_pose_xyz_m": list(document.get("task_pose_xyz_m", [])),
        "in_workspace_box": bool(document.get("in_workspace_box", True)),
    }
