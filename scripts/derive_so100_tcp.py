"""Transfer the official SO-101 TCP (gripper_frame_link) onto the SO-100 URDF.

The two arms share their axis geometry exactly (verified: the wrist_roll->gripper
common normal is 20.20 mm at 90 deg in both). So we can build a frame attached to
the gripper link purely from physical features -- the wrist_roll axis and the
gripper (jaw) axis -- which is identical on both robots regardless of how each
URDF names or orients its links. Expressing the SO-101 TCP in that frame yields
coordinates that transfer to SO-100 exactly.

Invariant gripper frame G:
    origin  = point on the gripper axis closest to the wrist_roll axis
    z       = gripper (jaw rotation) axis direction
    x       = common normal, pointing from the wrist_roll axis to the gripper axis
    y       = z cross x
"""

import os

UPSTREAM = os.environ.get("SOARM_UPSTREAM", "").rstrip("/")
if not UPSTREAM:
    raise SystemExit(
        "Set SOARM_UPSTREAM to a checkout of https://github.com/TheRobotStudio/SO-ARM100\n"
        "  git clone --depth 1 --filter=blob:none --sparse "
        "https://github.com/TheRobotStudio/SO-ARM100.git /tmp/soarm\n"
        "  (cd /tmp/soarm && git sparse-checkout set Simulation)\n"
        "  export SOARM_UPSTREAM=/tmp/soarm"
    )

import xml.etree.ElementTree as ET

import numpy as np

SO100 = f"{UPSTREAM}/Simulation/SO100/so100.urdf"
SO101 = f"{UPSTREAM}/Simulation/SO101/so101_new_calib.urdf"
ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def rpy_to_R(r, p, y):
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load(path):
    root = ET.parse(path).getroot()
    j = {}
    for e in root.findall("joint"):
        o = e.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        ax = e.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
        j[e.get("name")] = dict(xyz=xyz, rpy=rpy, axis=axis, parent=e.find("parent").get("link"),
                                child=e.find("child").get("link"))
    return j


def chain_frames(joints, order):
    """World transform of each joint's child-link frame, at zero configuration."""
    T = np.eye(4)
    frames = {}
    for n in order:
        j = joints[n]
        M = np.eye(4)
        M[:3, :3] = rpy_to_R(*j["rpy"])
        M[:3, 3] = j["xyz"]
        T = T @ M
        frames[n] = T.copy()
    return frames


def closest_points(p1, d1, p2, d2):
    """Closest points between two skew lines."""
    r = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ r, d2 @ r
    den = a * c - b * b
    if abs(den) < 1e-12:
        t1 = 0.0
        t2 = e / c
    else:
        t1 = (b * e - c * d) / den
        t2 = (a * e - b * d) / den
    return p1 + t1 * d1, p2 + t2 * d2


def invariant_gripper_frame(joints):
    """Frame G attached to the gripper link, built only from physical axis geometry."""
    fr = chain_frames(joints, ORDER)
    T_wr, T_gr = fr["wrist_roll"], fr["gripper"]
    p_wr, d_wr = T_wr[:3, 3], T_wr[:3, :3] @ joints["wrist_roll"]["axis"]
    p_gr, d_gr = T_gr[:3, 3], T_gr[:3, :3] @ joints["gripper"]["axis"]
    d_wr = d_wr / np.linalg.norm(d_wr)
    d_gr = d_gr / np.linalg.norm(d_gr)

    c_wr, c_gr = closest_points(p_wr, d_wr, p_gr, d_gr)
    z = d_gr
    x = c_gr - c_wr
    nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(z, x)

    G = np.eye(4)
    G[:3, 0], G[:3, 1], G[:3, 2] = x, y, z
    G[:3, 3] = c_gr
    return G, fr, nx


j100, j101 = load(SO100), load(SO101)
G100, fr100, n100 = invariant_gripper_frame(j100)
G101, fr101, n101 = invariant_gripper_frame(j101)

print(f"common normal wrist_roll->gripper:  SO-100 {n100 * 1000:.3f} mm   SO-101 {n101 * 1000:.3f} mm")

# SO-101's official TCP, in world coords at zero configuration
tcp = j101["gripper_frame_joint"]
T_gl = fr101["wrist_roll"]  # gripper_frame_joint's parent is gripper_link == child of wrist_roll
M = np.eye(4)
M[:3, :3] = rpy_to_R(*tcp["rpy"])
M[:3, 3] = tcp["xyz"]
T_tcp_world = T_gl @ M

# express it in the invariant frame G101
T_G_tcp = np.linalg.inv(G101) @ T_tcp_world
print("\nSO-101 TCP expressed in the invariant gripper frame G:")
print(f"  xyz (mm) = {np.array2string(T_G_tcp[:3, 3] * 1000, precision=3, floatmode='fixed')}")
print("  R =")
print(np.array2string(T_G_tcp[:3, :3], precision=5, suppress_small=True, prefix="    "))

# map it onto SO-100: same coordinates in G100, then back into the SO-100 gripper link frame
T_tcp_world_100 = G100 @ T_G_tcp
T_gripperlink_tcp = np.linalg.inv(fr100["wrist_roll"]) @ T_tcp_world_100

xyz = T_gripperlink_tcp[:3, 3]
R = T_gripperlink_tcp[:3, :3]
# recover rpy (ZYX convention, matching URDF)
pitch = np.arcsin(-np.clip(R[2, 0], -1, 1))
if abs(np.cos(pitch)) > 1e-8:
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
else:
    roll = np.arctan2(-R[1, 2], R[1, 1])
    yaw = 0.0

print("\n" + "=" * 70)
print("SO-100 gripper_frame_joint  (parent link: 'gripper')")
print("=" * 70)
print(f'  <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" '
      f'rpy="{roll:.6f} {pitch:.6f} {yaw:.6f}"/>')

# sanity: TCP distance from the gripper axis, must match between the two robots
d100 = np.linalg.norm(T_tcp_world_100[:3, 3] - G100[:3, 3])
d101 = np.linalg.norm(T_tcp_world[:3, 3] - G101[:3, 3])
print(f"\nsanity — TCP distance from gripper-axis origin: "
      f"SO-100 {d100 * 1000:.4f} mm vs SO-101 {d101 * 1000:.4f} mm")
