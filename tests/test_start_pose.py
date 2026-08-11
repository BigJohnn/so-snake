"""The recorded homing target: what may be written, and what may be read back.

Homing commands this file blind, at whatever speed the config says, with the
operator's hands off. Everything below is about the two moments that can put a
bad pose in front of that move -- writing one, and reading one somebody else
wrote or edited.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from so_snake.config import ARM_JOINTS, GRIPPER_JOINT, ArmConfig
from so_snake.start_pose import (
    JOINT_ORDER,
    StartPoseError,
    check_joint_limits,
    describe_start_pose,
    load_start_pose,
    save_start_pose,
)

ARM = ArmConfig()


def a_valid_pose() -> np.ndarray:
    """Mid-range everywhere: inside the limits by construction."""
    return np.array([sum(ARM.joint_limits_deg[j]) / 2.0 for j in JOINT_ORDER])


def test_joint_order_is_the_backend_order():
    """The file is read into a command array; a different order commands nonsense."""
    assert JOINT_ORDER == (*ARM_JOINTS, GRIPPER_JOINT)


def test_a_saved_pose_reads_back_as_the_same_joints(tmp_path):
    path = tmp_path / "start.json"
    pose = a_valid_pose()
    document = save_start_pose(pose, path=path, source="test")

    assert np.allclose(load_start_pose(path), pose, atol=1e-3)
    assert set(document["joints_urdf_deg"]) == set(JOINT_ORDER)
    # The frame has to travel with the numbers: lerobot's calibration frame is a
    # different set of degrees for the same physical pose.
    assert "URDF" in document["convention"]
    assert document["recorded_from"] == "test"


def test_saving_a_pose_outside_the_joint_limits_is_refused(tmp_path):
    """Recorded now, driven into a limit clamp later with nobody watching."""
    path = tmp_path / "start.json"
    pose = a_valid_pose()
    pose[0] = ARM.joint_limits_deg[ARM_JOINTS[0]][1] + 10.0

    with pytest.raises(StartPoseError, match="joint limits"):
        save_start_pose(pose, path=path)
    assert not path.exists()


def test_a_hand_edited_pose_outside_the_limits_is_refused_on_read(tmp_path):
    """The file is editable, and by whoever pulls the repo next -- check again."""
    path = tmp_path / "start.json"
    save_start_pose(a_valid_pose(), path=path)
    document = json.loads(path.read_text())
    document["joints_urdf_deg"][ARM_JOINTS[1]] = 999.0
    path.write_text(json.dumps(document))

    with pytest.raises(StartPoseError, match="outside the joint limits"):
        load_start_pose(path)


def test_a_pose_missing_a_joint_is_refused_rather_than_partially_used(tmp_path):
    path = tmp_path / "start.json"
    save_start_pose(a_valid_pose(), path=path)
    document = json.loads(path.read_text())
    del document["joints_urdf_deg"][GRIPPER_JOINT]
    path.write_text(json.dumps(document))

    with pytest.raises(StartPoseError, match="missing joints"):
        load_start_pose(path)


def test_no_file_is_not_an_error(tmp_path):
    """"No start pose recorded" is a normal state: homing falls back to config."""
    assert load_start_pose(tmp_path / "nothing.json") is None


def test_a_pose_outside_the_workspace_box_is_recorded_not_refused(tmp_path):
    """The box is the teleoperation clamp; starting folded outside it is normal.

    The bench's own start pose sits outside the box, so refusing here would
    reject a pose the operator deliberately chose.
    """
    path = tmp_path / "start.json"
    document = save_start_pose(a_valid_pose(), path=path, task_pose=[9.0, 9.0, 9.0, 0.0, 0.0])
    assert document["in_workspace_box"] is False
    assert load_start_pose(path) is not None


def test_describe_reports_the_reason_instead_of_raising(tmp_path):
    """A pose that cannot be used must be *shown*, not silently swapped out."""
    path = tmp_path / "start.json"
    path.write_text("{ this is not json")

    described = describe_start_pose(path)
    assert described["source"] == "config"
    assert "unreadable" in described["error"]


def test_describe_says_which_pose_homing_will_use(tmp_path):
    path = tmp_path / "start.json"
    save_start_pose(a_valid_pose(), path=path)

    described = describe_start_pose(path)
    assert described["source"] == "file"
    assert described["error"] == ""
    assert set(described["joints_deg"]) == set(JOINT_ORDER)


def test_check_joint_limits_names_every_joint_that_is_out(tmp_path):
    pose = a_valid_pose()
    pose[0] = 1000.0
    pose[2] = -1000.0
    problems = check_joint_limits(pose)
    assert len(problems) == 2
    assert ARM_JOINTS[0] in problems[0] and ARM_JOINTS[2] in problems[1]
