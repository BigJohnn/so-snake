#!/usr/bin/env python
"""Build the feasibility atlas and report what it says — no hardware required.

Samples the arm's joint space, forward-solves every sample, and bins the results
by workspace voxel, recording the pitch and roll each position was observed to
support. Writes `assets/atlas/so100_feasibility.npz`, which the teleoperation
loop loads to clamp pitch against something measured rather than assumed.

The summary at the end is the input to `TaskLimits.pitch_min_rad` /
`pitch_max_rad`: those must be the interval every voxel can hold, so that the
coarse clamp never forbids what the atlas would allow.

Re-run after any change to the URDF or to hardware calibration -- calibration
moves the reachable set, and a stale atlas is worse than none, because it looks
authoritative.

Run:  ./scripts/build_feasibility_atlas.py [--samples N] [--resolution M]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from so_snake.config import TaskLimits  # noqa: E402
from so_snake.kinematics import ArmChain  # noqa: E402
from so_snake.m3_safety.atlas import DEFAULT_ATLAS_PATH, FeasibilityAtlas  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4_000_000)
    parser.add_argument("--resolution", type=float, default=0.01, help="voxel edge, metres")
    parser.add_argument("--margin-deg", type=float, default=1.0, help="keep off the joint stops")
    parser.add_argument("--out", type=Path, default=DEFAULT_ATLAS_PATH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    chain = ArmChain()
    limits = TaskLimits()

    print(f"position box   x {limits.pos_min_m[0]:.3f}..{limits.pos_max_m[0]:.3f}  "
          f"y {limits.pos_min_m[1]:+.3f}..{limits.pos_max_m[1]:+.3f}  "
          f"z {limits.pos_min_m[2]:.3f}..{limits.pos_max_m[2]:.3f}  m")
    print(f"sampling {args.samples:,} configurations at {args.resolution * 1000:.0f} mm resolution\n")

    started = time.perf_counter()
    atlas = FeasibilityAtlas.build(
        chain,
        limits,
        resolution_m=args.resolution,
        samples=args.samples,
        margin_deg=args.margin_deg,
        progress=not args.quiet,
    )
    elapsed = time.perf_counter() - started

    path = atlas.save(args.out)
    print(f"\nbuilt in {elapsed:.1f} s -> {path} ({path.stat().st_size / 1e6:.2f} MB)")

    print("\n" + "=" * 72)
    print("COVERAGE")
    print("=" * 72)
    print(f"  grid            {atlas.shape[0]} x {atlas.shape[1]} x {atlas.shape[2]}"
          f" = {atlas.count.size} voxels")
    print(f"  reached         {int((atlas.count > 0).sum())} ({100 * atlas.coverage:.1f}%)")
    for threshold in (1, 10, 100, 1000):
        share = float((atlas.count >= threshold).mean())
        print(f"  >= {threshold:>5} samples  {100 * share:5.1f}%")

    print("\n" + "=" * 72)
    print("PITCH — elevation of the approach axis, per voxel")
    print("=" * 72)
    occupied = atlas.count > 0
    lows = np.degrees(atlas.pitch_min[occupied])
    highs = np.degrees(atlas.pitch_max[occupied])
    spans = highs - lows
    print(f"  lowest  pitch reached anywhere   {lows.min():+7.2f} deg")
    print(f"  highest pitch reached anywhere   {highs.max():+7.2f} deg")
    print(f"  per-voxel span   median {np.median(spans):6.2f}   p05 {np.percentile(spans, 5):6.2f}"
          f"   min {spans.min():6.2f} deg")

    for min_count in (1, 10, 100, 1000):
        try:
            low, high = atlas.pitch_envelope(min_count=min_count)
        except ValueError:
            print(f"  envelope over voxels with >= {min_count:>5} samples:  none that well sampled")
            continue
        universal = atlas.pitch_universal(min_count=min_count)
        universal_text = (
            "empty" if universal is None
            else f"[{np.degrees(universal[0]):+6.2f}, {np.degrees(universal[1]):+6.2f}]"
        )
        print(f"  >= {min_count:>5} samples   envelope [{np.degrees(low):+7.2f},"
              f" {np.degrees(high):+7.2f}] deg   available everywhere: {universal_text}")

    print("\n  -> TaskLimits.pitch_min_rad / pitch_max_rad take the envelope, rounded inward.")
    print("     'available everywhere' is empty by design: at the edge of reach the arm is")
    print("     nearly straight and pitch is pinned, which is exactly what the atlas is for.")

    # How much of the box can hold a near-vertical approach, which is what a
    # top-down grasp of a block on a table needs.
    for threshold_deg in (-90.0, -85.0, -80.0, -75.0, -70.0):
        share = float(
            (atlas.pitch_min[occupied] <= np.radians(threshold_deg)).mean()
        )
        print(f"  voxels able to point at least {threshold_deg:+.0f} deg down: {100 * share:5.1f}%")

    print("\n" + "=" * 72)
    print("ROLL — occupancy of 16 arcs of the circle, per voxel")
    print("=" * 72)
    occupancy = np.array([bin(int(v)).count("1") for v in atlas.roll_bins[occupied]])
    print(f"  arcs available   median {np.median(occupancy):.0f}/16"
          f"   p05 {np.percentile(occupancy, 5):.0f}/16   min {occupancy.min()}/16")
    print(f"  voxels with all 16 arcs   {100 * float((occupancy == 16).mean()):.1f}%")
    print("  (the URDF has no cabling, wrist-camera lead or self-collision geometry,")
    print("   so this overestimates roll freedom until measured on hardware)")

    print("\n" + "=" * 72)
    print("CONDITIONING — best smallest singular value of J per voxel")
    print("=" * 72)
    sigma = atlas.sigma_min_best[occupied]
    print(f"  median {np.median(sigma):.4f}   p05 {np.percentile(sigma, 5):.4f}"
          f"   min {sigma.min():.4f}")
    print(f"  voxels whose best configuration is still under the solver's"
          f" 0.01 damping threshold: {int((sigma < 0.01).sum())}")

    print("\n" + "=" * 72)
    print("YAW — is it really determined by position?")
    print("=" * 72)
    index = np.argwhere(atlas.count >= 100)
    centres = atlas.origin_m + (index + 0.5) * atlas.resolution_m
    measured = np.array([atlas.yaw_at(tuple(i)) for i in map(tuple, index)])
    # The pan axis sits at x = +0.0452 in the world frame, so the azimuth of a
    # voxel has to be measured from there, not from the origin.
    predicted = np.arctan2(centres[:, 1], centres[:, 0] - 0.0452)
    residual = np.degrees(np.abs((measured - predicted + np.pi) % (2 * np.pi) - np.pi))
    print(f"  |mean yaw - atan2(y, x - 0.0452)|   median {np.median(residual):5.2f}"
          f"   p95 {np.percentile(residual, 95):5.2f}   max {residual.max():5.2f} deg")
    print("  (a small residual is what licenses a closed-form R_position_yaw(p);")
    print("   the solver does not need one, so this is validation, not a dependency)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
