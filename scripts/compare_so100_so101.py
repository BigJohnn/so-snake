"""Fully frame-convention-invariant comparison of SO-100 vs SO-101.

Compares the geometry between consecutive joint AXES, treated as lines in 3D at
zero configuration. The common-normal distance and the twist angle between two
consecutive axes (the DH parameters a and alpha) are invariant under any choice
of link frames, so any difference here is genuinely mechanical.
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

A = f"{UPSTREAM}/Simulation/SO100/so100.urdf"
B = f"{UPSTREAM}/Simulation/SO101/so101_new_calib.urdf"
ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def rpy_to_R(r, p, y):
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axes_in_base(path):
    """Return each joint's rotation axis as (point_on_axis, unit_direction) in the base frame."""
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
        joints[j.get("name")] = (xyz, rpy, axis, j.find("parent").get("link"), j.find("child").get("link"))

    out = []
    T = np.eye(4)
    for n in ORDER:
        xyz, rpy, axis, _, _ = joints[n]
        M = np.eye(4)
        M[:3, :3] = rpy_to_R(*rpy)
        M[:3, 3] = xyz
        T = T @ M  # zero configuration: joint rotation contributes identity
        point = T[:3, 3]
        direction = T[:3, :3] @ axis
        direction = direction / np.linalg.norm(direction)
        out.append((point, direction))
    return out


def axis_pair_geometry(p1, d1, p2, d2):
    """Common-normal distance and twist angle between two lines in 3D."""
    n = np.cross(d1, d2)
    nn = np.linalg.norm(n)
    if nn < 1e-9:  # parallel axes
        delta = p2 - p1
        dist = np.linalg.norm(delta - np.dot(delta, d1) * d1)
    else:
        dist = abs(np.dot(p2 - p1, n / nn))
    ang = np.degrees(np.arccos(np.clip(abs(np.dot(d1, d2)), -1, 1)))
    return dist, ang


a = axes_in_base(A)
b = axes_in_base(B)

print("=" * 78)
print("DH-INVARIANT AXIS GEOMETRY (zero config) — independent of frame conventions")
print("=" * 78)
print(f"{'axis pair':<30}{'SO-100':>20}{'SO-101':>20}{'delta':>10}")
print(f"{'':<30}{'dist(mm) / twist':>20}{'dist(mm) / twist':>20}")
for i in range(len(ORDER) - 1):
    da, aa = axis_pair_geometry(*a[i], *a[i + 1])
    db, ab = axis_pair_geometry(*b[i], *b[i + 1])
    pair = f"{ORDER[i]} -> {ORDER[i + 1]}"
    flag = "  <-- DIFF" if abs(db - da) * 1000 > 0.5 or abs(ab - aa) > 0.5 else "  same"
    print(
        f"{pair:<30}"
        f"{f'{da * 1000:8.2f} / {aa:6.2f}deg':>20}"
        f"{f'{db * 1000:8.2f} / {ab:6.2f}deg':>20}"
        f"{f'{(db - da) * 1000:+.2f}mm':>10}{flag}"
    )

print()
print("=" * 78)
print("SHOULDER-PAN AXIS -> WRIST-ROLL AXIS  (overall arm span, invariant)")
print("=" * 78)
for label, chain in (("SO-100", a), ("SO-101", b)):
    d, ang = axis_pair_geometry(*chain[0], *chain[4])
    print(f"{label}: common-normal distance {d * 1000:7.2f} mm, twist {ang:6.2f} deg")
