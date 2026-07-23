#!/usr/bin/env python
"""Generate the MuJoCo model from the SO-100 URDF — no hardware required.

Writes `assets/mujoco/so100.xml`, the scene `so_snake.sim` loads. Generated
rather than hand-written so that the URDF stays the single source of truth for
the arm's geometry: change a link, re-run this, and the simulation follows.

Four things the URDF cannot express are added here:

  * **TCP and tool sites.** MuJoCo folds massless fixed-joint links into their
    parent, so `gripper_frame_link` does not survive the import. It comes back
    as a site, and a second site carries the tool convention of
    `ArmChain.tool_from_tcp` -- +X along the approach axis -- so that the
    simulation reports the same frame the task space is defined in.
  * **The world frame.** The URDF base has the arm extending along -Y. The base
    body is rotated by +90 deg about Z so that MuJoCo's world frame is the
    +X-forward world frame everything else uses, and a pose read out of the
    simulation can be compared with one from `ArmChain` without a conversion
    that could itself be wrong.
  * **Cameras.** A wrist camera on the gripper and a third-person camera, the
    two the blueprint records from. Having them in simulation is how the
    recording pipeline and the policy's observation layout get exercised before
    any USB camera is plugged in.
  * **A table and lighting**, so rendered frames look like the workspace rather
    than like a robot floating in a void.

Run:  ./scripts/build_mujoco_model.py
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from so_snake.config import ArmConfig, TeleopConfig  # noqa: E402
from so_snake.kinematics import ArmChain, _parse_joints  # noqa: E402

DEFAULT_OUTPUT = REPO / "assets" / "mujoco" / "so100.xml"

# Where the wrist camera sits on the gripper body, in that body's own frame,
# and which way it faces. Set back along +Y from the jaws (the gripper's -Y is
# the approach direction) and clear of them along +Z. Chosen by rendering:
# further forward puts the camera inside the jaw mesh, and further back loses
# the jaws out of frame.
WRIST_CAMERA_POS = (0.0, 0.03, 0.035)
# MuJoCo cameras look down their own -Z with +Y up in the image. `xyaxes` gives
# the camera's X and Y: X = -X_gripper and Y = +Z_gripper put +Z_camera on the
# gripper's +Y, so the camera looks along the approach direction with the jaws
# entering frame from the bottom, the way a wrist camera is mounted.
WRIST_CAMERA_XYAXES = "-1 0 0 0 0 1"
WRIST_CAMERA_FOVY = 75.0

THIRD_PERSON_POS = (0.62, -0.52, 0.45)
THIRD_PERSON_FOVY = 45.0


def _matrix_to_quat(R: np.ndarray) -> str:
    """MuJoCo's `w x y z` quaternion string for a rotation matrix."""
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
        q = np.zeros(3)
        q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
        w = (R[k, j] - R[j, k]) / s
        x, y, z = q
    return " ".join(f"{v:.9g}" for v in (w, x, y, z))


def _export_raw_mjcf(urdf_path: Path, destination: Path) -> None:
    """Let MuJoCo do the URDF parsing, then take its own XML back out.

    Round-tripping through the compiler rather than translating the URDF by hand
    means the geometry in the MJCF is exactly what MuJoCo would have loaded,
    with none of the conventions reimplemented and none of them silently wrong.
    """
    import mujoco

    tree = ET.parse(urdf_path)
    extension = ET.SubElement(tree.getroot(), "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        meshdir=".",
        # MuJoCo strips directories from URDF mesh paths by default, which
        # breaks `assets/Base.stl`; and it fuses static bodies away, which would
        # take the base link with it.
        strippath="false",
        fusestatic="false",
        balanceinertia="true",
        discardvisual="false",
    )

    staged = urdf_path.parent / "_mujoco_export.urdf"
    tree.write(staged)
    try:
        model = mujoco.MjModel.from_xml_path(str(staged))
        destination.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(destination), model)
    finally:
        staged.unlink(missing_ok=True)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"body {name!r} not found in the generated MJCF")


def build(urdf_path: Path, output: Path) -> ET.ElementTree:
    arm = ArmConfig()
    chain = ArmChain(arm)
    joints = _parse_joints(arm.urdf_path)

    _export_raw_mjcf(urdf_path, output)
    tree = ET.parse(output)
    root = tree.getroot()
    root.set("model", "so100_snake")

    # Mesh paths in the exported MJCF are relative to the URDF's directory, but
    # the MJCF lands somewhere else, so `meshdir` has to bridge the two.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(Path(os.path.relpath(urdf_path.parent, output.parent))))
    compiler.set("angle", "radian")

    worldbody = root.find("worldbody")
    base = _find_body(worldbody, chain.root_link)

    # --- world frame -------------------------------------------------------
    base.set("quat", _matrix_to_quat(arm.world_from_base()[:3, :3]))

    # --- TCP and tool sites ------------------------------------------------
    tcp_joint = joints[f"{arm.tcp_frame.removesuffix('_link')}_joint"]
    gripper_body = _find_body(worldbody, tcp_joint.parent)
    position = " ".join(f"{v:.9g}" for v in tcp_joint.origin[:3, 3])
    ET.SubElement(
        gripper_body,
        "site",
        name="tcp",
        pos=position,
        quat=_matrix_to_quat(tcp_joint.origin[:3, :3]),
        size="0.004",
        rgba="1 0.2 0.2 1",
        group="3",
    )
    ET.SubElement(
        gripper_body,
        "site",
        name="tool",
        pos=position,
        quat=_matrix_to_quat(tcp_joint.origin[:3, :3] @ chain.tool_from_tcp()),
        size="0.004",
        rgba="0.2 1 0.2 1",
        group="3",
    )

    # --- cameras -----------------------------------------------------------
    ET.SubElement(
        gripper_body,
        "camera",
        name="wrist",
        pos=" ".join(str(v) for v in WRIST_CAMERA_POS),
        xyaxes=WRIST_CAMERA_XYAXES,
        fovy=str(WRIST_CAMERA_FOVY),
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="third_person",
        pos=" ".join(str(v) for v in THIRD_PERSON_POS),
        mode="targetbody",
        target=chain.root_link,
        fovy=str(THIRD_PERSON_FOVY),
    )

    # --- scene -------------------------------------------------------------
    asset = root.find("asset")
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0.24 0.26 0.29",
        rgb2="0.28 0.30 0.33",
        width="300",
        height="300",
    )
    ET.SubElement(
        asset, "material", name="table", texture="grid", texrepeat="8 8", reflectance="0.05"
    )
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.16 0.18 0.22", rgb2="0.04 0.05 0.07", width="256", height="256")

    # The table is where the arm's base is bolted, so it sits at z = 0 and the
    # base's own mesh rests just above it. Contact with the table is excluded
    # below rather than modelled: this is a kinematic simulation, and a resting
    # contact would show up as a permanent false positive in collision checks.
    ET.SubElement(
        worldbody,
        "geom",
        name="table",
        type="plane",
        size="1.5 1.5 0.01",
        material="table",
        contype="0",
        conaffinity="0",
        group="2",
    )
    ET.SubElement(worldbody, "light", name="key", pos="0.4 -0.6 1.2", dir="-0.2 0.4 -1",
                  directional="true", diffuse="0.7 0.7 0.7")
    ET.SubElement(worldbody, "light", name="fill", pos="-0.4 0.6 0.9", dir="0.3 -0.4 -1",
                  directional="true", diffuse="0.25 0.25 0.28")

    visual = root.find("visual") or ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")

    # --- neighbouring links always touch; only distant ones are informative --
    ordered = [chain.root_link, *(joints[name].child for name in arm.joint_names)]
    contact = ET.SubElement(root, "contact")
    for first, second in zip(ordered, ordered[1:], strict=False):
        ET.SubElement(contact, "exclude", body1=first, body2=second)

    # --- home keyframe -----------------------------------------------------
    home = np.deg2rad(np.array(TeleopConfig().home_joints_deg, dtype=float))
    gripper_mid = np.deg2rad(sum(arm.joint_limits_deg["gripper"]) / 2.0)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="home",
        qpos=" ".join(f"{v:.9g}" for v in [*home, gripper_mid]),
    )

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return tree


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    arm = ArmConfig()
    build(arm.urdf_path, args.out)
    print(f"URDF   {arm.urdf_path}")
    print(f"model  {args.out}  ({args.out.stat().st_size / 1024:.1f} kB)")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.out))
    print(f"\nloaded: {model.nq} joints, {model.nbody} bodies, {model.ngeom} geoms, "
          f"{model.ncam} cameras, {model.nkey} keyframes")
    for i in range(model.nsite):
        print(f"  site   {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")
    for i in range(model.ncam):
        print(f"  camera {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
