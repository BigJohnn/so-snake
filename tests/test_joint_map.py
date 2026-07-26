"""The lerobot<->URDF joint map must be an exact bijection, or the arm jumps."""

import json

import numpy as np
import pytest

from so_snake.config import ARM_JOINTS, GRIPPER_JOINT, JOINT_LIMITS_DEG
from so_snake.m4_execution import JointFrameMap


def _sample_map(signs: dict[str, int] | None = None) -> JointFrameMap:
    signs = signs or {j: 1 for j in ARM_JOINTS}
    joints = {}
    for name in ARM_JOINTS:
        lo, hi = JOINT_LIMITS_DEG[name]
        joints[name] = {"type": "degrees_affine", "sign": signs[name], "offset_deg": (lo + hi) / 2.0}
    glo, ghi = JOINT_LIMITS_DEG[GRIPPER_JOINT]
    joints[GRIPPER_JOINT] = {"type": "range_0_100_to_deg", "urdf_min_deg": glo, "urdf_max_deg": ghi}
    return JointFrameMap(joints=joints, arm_joints=ARM_JOINTS, gripper_joint=GRIPPER_JOINT)


@pytest.mark.parametrize("sign", [1, -1])
def test_affine_roundtrips_exactly(sign):
    m = _sample_map({j: sign for j in ARM_JOINTS})
    for name in ARM_JOINTS:
        for lero in np.linspace(-120, 120, 25):
            urdf = m.lerobot_to_urdf(name, lero)
            assert m.urdf_to_lerobot(name, urdf) == pytest.approx(lero, abs=1e-9)


def test_sign_flips_direction():
    pos = _sample_map({j: 1 for j in ARM_JOINTS})
    neg = _sample_map({j: -1 for j in ARM_JOINTS})
    # A positive step in lerobot degrees maps to opposite URDF directions.
    name = "elbow_flex"
    off = (JOINT_LIMITS_DEG[name][0] + JOINT_LIMITS_DEG[name][1]) / 2.0
    assert pos.lerobot_to_urdf(name, 10.0) - off == pytest.approx(+10.0)
    assert neg.lerobot_to_urdf(name, 10.0) - off == pytest.approx(-10.0)


def test_gripper_roundtrips_within_range():
    m = _sample_map()
    for lero in np.linspace(0, 100, 21):
        urdf = m.lerobot_to_urdf(GRIPPER_JOINT, lero)
        assert m.urdf_to_lerobot(GRIPPER_JOINT, urdf) == pytest.approx(lero, abs=1e-9)
    glo, ghi = JOINT_LIMITS_DEG[GRIPPER_JOINT]
    assert m.lerobot_to_urdf(GRIPPER_JOINT, 0.0) == pytest.approx(glo)
    assert m.lerobot_to_urdf(GRIPPER_JOINT, 100.0) == pytest.approx(ghi)


def test_load_from_json(tmp_path):
    m = _sample_map()
    data = {
        "joints": m.joints,
        "arm_joints": list(ARM_JOINTS),
        "gripper_joint": GRIPPER_JOINT,
    }
    path = tmp_path / "map.json"
    path.write_text(json.dumps(data))
    loaded = JointFrameMap.load(path)
    assert loaded.arm_joints == ARM_JOINTS
    assert loaded.lerobot_to_urdf("shoulder_pan", 12.3) == pytest.approx(m.lerobot_to_urdf("shoulder_pan", 12.3))
