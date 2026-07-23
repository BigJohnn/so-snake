#!/usr/bin/env python
"""Three independent kinematics implementations must agree — no hardware required.

`ArmChain` is the control path's forward kinematics and Jacobian. It is our own
code, so its tests are also our own code, and a shared misreading of the URDF
would pass all of them. placo (via lerobot) and MuJoCo parse the same URDF with
no shared lineage, so agreement between the three is evidence that the frame
conventions, the joint order, the TCP offset and the world rotation are right.

Checks, over configurations drawn from the whole joint space:

  1. TCP position and orientation:  ArmChain  vs  placo  vs  MuJoCo
  2. Geometric Jacobian:            ArmChain  vs  MuJoCo's `mj_jacSite`
  3. The tool frame the task space is defined in, on both sides

Run:  ./scripts/check_kinematics_agreement.py [--samples N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from so_snake.config import ArmConfig  # noqa: E402
from so_snake.kinematics import ArmChain  # noqa: E402
from so_snake.sim import MujocoArm  # noqa: E402


def _report(label: str, values: np.ndarray, unit: str, tolerance: float) -> bool:
    worst = float(np.max(values))
    ok = worst <= tolerance
    print(f"  {label:<44} max {worst:9.3e} {unit:<6} {'OK' if ok else 'FAIL'}"
          f"  (tolerance {tolerance:.0e})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    arm = ArmConfig()
    chain = ArmChain(arm)
    sim = MujocoArm(arm=arm)
    R_tcp_tool = chain.tool_from_tcp()

    from lerobot.model.kinematics import RobotKinematics

    placo = RobotKinematics(
        str(arm.urdf_path), target_frame_name=arm.tcp_frame, joint_names=list(arm.joint_names)
    )
    world_from_base = arm.world_from_base()

    lo, hi = arm.limits_deg_array()
    rng = np.random.default_rng(args.seed)
    configurations = rng.uniform(lo, hi, size=(args.samples, len(arm.joint_names)))

    placo_position, placo_rotation = [], []
    mujoco_position, mujoco_rotation = [], []
    mujoco_tool, mujoco_jacobian = [], []

    for q in configurations:
        ours = chain.fk(q)

        theirs = world_from_base @ placo.forward_kinematics(q)
        placo_position.append(np.abs(ours[:3, 3] - theirs[:3, 3]).max())
        placo_rotation.append(np.abs(ours[:3, :3] - theirs[:3, :3]).max())

        sim.set_joints_deg(q)
        simulated = sim.tcp_pose()
        mujoco_position.append(np.abs(ours[:3, 3] - simulated[:3, 3]).max())
        mujoco_rotation.append(np.abs(ours[:3, :3] - simulated[:3, :3]).max())

        tool = sim.tool_pose()
        mujoco_tool.append(np.abs(ours[:3, :3] @ R_tcp_tool - tool[:3, :3]).max())
        mujoco_jacobian.append(np.abs(chain.jacobian(q) - sim.jacobian()).max())

    print(f"URDF     {arm.urdf_path}")
    print(f"MuJoCo   {sim.model_path}")
    print(f"samples  {args.samples} configurations over the full joint ranges\n")

    print("=" * 78)
    print("AGREEMENT")
    print("=" * 78)
    # placo shares our double precision and agrees to machine epsilon. MuJoCo
    # stores body frames as single-precision quaternions, so its rotations carry
    # about 1e-6 of representation error -- roughly 1e-4 degrees, or 30 nm of
    # arc at the tool. Tolerances say which of the two is being compared; a
    # regression that mattered would be orders of magnitude larger than either.
    checks = [
        _report("TCP position   ArmChain vs placo", np.array(placo_position), "m", 1e-9),
        _report("TCP rotation   ArmChain vs placo", np.array(placo_rotation), "", 1e-9),
        _report("TCP position   ArmChain vs MuJoCo", np.array(mujoco_position), "m", 1e-5),
        _report("TCP rotation   ArmChain vs MuJoCo", np.array(mujoco_rotation), "", 1e-5),
        _report("tool rotation  ArmChain vs MuJoCo", np.array(mujoco_tool), "", 1e-5),
        _report("Jacobian       ArmChain vs MuJoCo", np.array(mujoco_jacobian), "", 1e-5),
    ]

    if all(checks):
        print("\nAll three implementations agree.")
        return 0
    print("\nDISAGREEMENT — do not trust the control path until this is understood.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
