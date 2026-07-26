"""The per-joint frame map between lerobot-calibrated degrees and URDF degrees.

so-snake's kinematics/IK/safety all live in the official SO-ARM100 URDF frame;
lerobot's `SOFollower` reads and accepts degrees in its own calibration frame
(zero at the recorded range midpoint, direction set by servo mounting). The two
differ, per joint, by an affine map with unit gain:

    q_urdf = sign * q_lerobot + offset      sign in {+1, -1}      (arm joints)

The gripper is reported by lerobot on 0..100, not in degrees, so it maps linearly
onto the URDF gripper range instead.

Produced and checked by `scripts/map_joint_frames.py`. `SOFollowerBackend` applies
it so the control loop only ever sees URDF degrees. The map is an exact bijection,
so a value read and written straight back round-trips to itself -- which is what
keeps the arm from jumping on the first command even when `offset` is only
approximate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JointFrameMap:
    joints: dict
    arm_joints: tuple[str, ...]
    gripper_joint: str

    @classmethod
    def load(cls, path: str | Path) -> "JointFrameMap":
        data = json.loads(Path(path).read_text())
        missing = [n for n in (*data["arm_joints"], data["gripper_joint"]) if n not in data["joints"]]
        if missing:
            raise ValueError(f"joint map {path} is missing entries for {missing}")
        return cls(
            joints=data["joints"],
            arm_joints=tuple(data["arm_joints"]),
            gripper_joint=data["gripper_joint"],
        )

    def lerobot_to_urdf(self, name: str, value: float) -> float:
        j = self.joints[name]
        if j["type"] == "degrees_affine":
            return j["sign"] * value + j["offset_deg"]
        lo, hi = j["urdf_min_deg"], j["urdf_max_deg"]
        frac = min(100.0, max(0.0, value)) / 100.0
        return lo + frac * (hi - lo)

    def urdf_to_lerobot(self, name: str, value: float) -> float:
        j = self.joints[name]
        if j["type"] == "degrees_affine":
            # sign in {+1,-1}, so 1/sign == sign: lero = (urdf - offset) / sign.
            return j["sign"] * (value - j["offset_deg"])
        lo, hi = j["urdf_min_deg"], j["urdf_max_deg"]
        frac = (value - lo) / (hi - lo)
        return min(100.0, max(0.0, frac * 100.0))
