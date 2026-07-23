"""M3 — the feasibility atlas (plan step 6).

A box says which positions are reachable. It cannot say which *pitches* are
reachable at a given position, and that varies enormously: directly in front of
the base the gripper can be tilted through most of a right angle, while at the
edge of reach the arm is nearly straight and pitch is pinned to within a few
degrees. Clamping pitch against one global interval would either forbid poses
the arm can hold, or permit poses it cannot and leave the IK to fail silently.

So the reachable set is measured once, offline, and stored as a voxel grid:

    sample joint space -> forward kinematics -> bin by XYZ voxel
    -> per voxel record what orientations were seen there

The online path is then `5D target -> box clamp -> atlas projection -> 5D IK ->
joint safety`, with the atlas step turning "the operator asked for something
unreachable" into "the operator hit a wall", which is a thing they can feel and
back away from.

Sampling forward rather than solving IK per voxel is deliberate. Forward
kinematics cannot fail, cannot land in a local minimum, and cannot be wrong
about reachability -- every sample is a configuration the arm demonstrably
holds. The cost is coverage: a voxel with few samples is undersampled, not
unreachable, and `count` is stored so the difference stays visible.

Roll is stored as a bin mask rather than an interval because it is periodic:
`wrist_roll` spans the full circle, so "min to max" across samples degenerates
to "-180 to +180" and says nothing. A 16-bin occupancy mask says what an
interval cannot -- which arcs of the circle are actually available here.

Caveats that hardware will change, and that no amount of URDF sampling can fix:
the URDF has no cabling, no wrist-camera lead, no self-collision geometry and no
TCP mount, so the atlas **overestimates** roll freedom in particular. Re-run it
after any calibration change; calibration moves the reachable set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import REPO_ROOT, TaskLimits
from ..kinematics import ArmChain
from .task_pose import SO100TaskPose, TaskFrame

DEFAULT_ATLAS_PATH = REPO_ROOT / "assets" / "atlas" / "so100_feasibility.npz"

ROLL_BINS = 16


@dataclass
class AtlasProjection:
    """What the atlas did to a requested task pose."""

    pose: SO100TaskPose
    pitch_clamped: bool
    roll_infeasible: bool  # requested roll fell in an unoccupied arc; left as asked
    voxel_count: int  # samples behind this cell; 0 means the cell was never reached
    outside_atlas: bool  # the position had to be snapped to a populated cell

    @property
    def any_clamped(self) -> bool:
        return bool(self.pitch_clamped or self.outside_atlas)


class FeasibilityAtlas:
    """Per-voxel record of the orientations the arm was observed to reach."""

    def __init__(
        self,
        origin_m: np.ndarray,
        resolution_m: float,
        shape: tuple[int, int, int],
        count: np.ndarray,
        pitch_min: np.ndarray,
        pitch_max: np.ndarray,
        roll_bins: np.ndarray,
        yaw_sin: np.ndarray,
        yaw_cos: np.ndarray,
        sigma_min_best: np.ndarray,
        limit_margin_best: np.ndarray,
        metadata: dict[str, float] | None = None,
    ) -> None:
        self.origin_m = np.asarray(origin_m, float)
        self.resolution_m = float(resolution_m)
        self.shape = tuple(shape)
        self.count = count
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.roll_bins = roll_bins
        self.yaw_sin = yaw_sin
        self.yaw_cos = yaw_cos
        self.sigma_min_best = sigma_min_best
        self.limit_margin_best = limit_margin_best
        self.metadata = dict(metadata or {})

        self._occupied = self.count > 0
        occupied_index = np.argwhere(self._occupied)
        self._occupied_index = occupied_index
        self._occupied_centres = self.origin_m + (occupied_index + 0.5) * self.resolution_m

    # ---------------------------------------------------------------- building

    @classmethod
    def build(
        cls,
        chain: ArmChain | None = None,
        limits: TaskLimits | None = None,
        *,
        resolution_m: float = 0.01,
        samples: int = 4_000_000,
        chunk: int = 250_000,
        seed: int = 0,
        margin_deg: float = 0.0,
        progress: bool = False,
    ) -> FeasibilityAtlas:
        """Sample joint space and bin the results.

        Args:
            resolution_m: Voxel edge. 10 mm is a little under the 30 mm block the
                task grasps, so a cell is small enough that its pitch interval
                still means something at the scale of the object.
            samples: Total joint configurations drawn. Coverage, not accuracy,
                is what this buys: every sample is exact.
            margin_deg: Shrink the joint ranges by this much before sampling, to
                keep the atlas away from the mechanical stops.
        """
        chain = chain or ArmChain()
        limits = limits or TaskLimits()
        arm = chain.arm

        lo, hi = arm.limits_deg_array()
        lo, hi = lo + margin_deg, hi - margin_deg

        origin = np.asarray(limits.pos_min_m, float)
        span = np.asarray(limits.pos_max_m, float) - origin
        shape = tuple(int(np.ceil(s / resolution_m)) for s in span)

        count = np.zeros(shape, dtype=np.int64)
        pitch_min = np.full(shape, np.inf)
        pitch_max = np.full(shape, -np.inf)
        roll_bins = np.zeros(shape, dtype=np.uint16)
        yaw_sin = np.zeros(shape)
        yaw_cos = np.zeros(shape)
        sigma_best = np.zeros(shape)
        margin_best = np.zeros(shape)

        R_tcp_tool = chain.tool_from_tcp()
        frame = TaskFrame(chain)
        rng = np.random.default_rng(seed)
        half_range = (hi - lo) / 2.0

        drawn = 0
        while drawn < samples:
            n = min(chunk, samples - drawn)
            drawn += n
            q = rng.uniform(lo, hi, size=(n, chain.n_joints))

            T = chain.fk_batch(q)
            position = T[:, :3, 3]

            index = np.floor((position - origin) / resolution_m).astype(np.int64)
            inside = np.all((index >= 0) & (index < np.array(shape)), axis=1)
            if not inside.any():
                continue

            q, T, index = q[inside], T[inside], index[inside]
            flat = np.ravel_multi_index(tuple(index.T), shape)

            M = T[:, :3, :3] @ R_tcp_tool
            u = M[:, :, 0]

            # The chart's angles, batched. `psi` comes from position, so the
            # rotation into the plane frame is `Rz(-psi) M` -- and only five of
            # its entries are needed, so they are formed directly.
            psi = np.arctan2(
                position[:, 1] - frame.pan_axis_xy[1], position[:, 0] - frame.pan_axis_xy[0]
            )
            c, s = np.cos(psi), np.sin(psi)
            pitch = np.arctan2(M[:, 2, 0], c * M[:, 0, 0] + s * M[:, 1, 0])
            roll = np.arctan2(
                -(-s * M[:, 0, 2] + c * M[:, 1, 2]), -s * M[:, 0, 1] + c * M[:, 1, 1]
            )

            # Measured tool azimuth, kept so that `psi(p)` can be validated
            # against what the arm actually holds. See the yaw section of
            # `scripts/build_feasibility_atlas.py`.
            yaw = np.arctan2(u[:, 1], u[:, 0])

            # Smallest singular value of the 6x5 Jacobian: how close this
            # configuration is to losing a degree of freedom outright. The
            # solver's own `J_task` is a projection of it, so this bounds it.
            sigma = np.linalg.svd(chain.jacobian_batch(q), compute_uv=False)[:, -1]
            margin = np.min(np.minimum(q - lo, hi - q), axis=1)

            roll_index = np.floor((roll + np.pi) / (2 * np.pi) * ROLL_BINS).astype(np.int64)
            roll_bit = (1 << (roll_index % ROLL_BINS)).astype(np.uint16)

            flat_count = count.reshape(-1)
            np.add.at(flat_count, flat, 1)
            np.minimum.at(pitch_min.reshape(-1), flat, pitch)
            np.maximum.at(pitch_max.reshape(-1), flat, pitch)
            np.bitwise_or.at(roll_bins.reshape(-1), flat, roll_bit)
            np.add.at(yaw_sin.reshape(-1), flat, np.sin(yaw))
            np.add.at(yaw_cos.reshape(-1), flat, np.cos(yaw))
            np.maximum.at(sigma_best.reshape(-1), flat, sigma)
            np.maximum.at(margin_best.reshape(-1), flat, margin)

            if progress:
                filled = int((count > 0).sum())
                print(
                    f"  {drawn:>10,} samples   {filled:>6}/{count.size} voxels "
                    f"({100.0 * filled / count.size:5.1f}%)",
                    flush=True,
                )

        empty = count == 0
        pitch_min[empty] = np.nan
        pitch_max[empty] = np.nan

        return cls(
            origin_m=origin,
            resolution_m=resolution_m,
            shape=shape,
            count=count,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            roll_bins=roll_bins,
            yaw_sin=yaw_sin,
            yaw_cos=yaw_cos,
            sigma_min_best=sigma_best,
            limit_margin_best=margin_best,
            metadata={
                "samples": float(samples),
                "seed": float(seed),
                "margin_deg": float(margin_deg),
            },
        )

    # ------------------------------------------------------------------- io

    def save(self, path: Path = DEFAULT_ATLAS_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            origin_m=self.origin_m,
            resolution_m=self.resolution_m,
            count=self.count,
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            roll_bins=self.roll_bins,
            yaw_sin=self.yaw_sin,
            yaw_cos=self.yaw_cos,
            sigma_min_best=self.sigma_min_best,
            limit_margin_best=self.limit_margin_best,
            metadata_keys=np.array(list(self.metadata), dtype=object),
            metadata_values=np.array(list(self.metadata.values()), dtype=float),
        )
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_ATLAS_PATH) -> FeasibilityAtlas:
        data = np.load(Path(path), allow_pickle=True)
        metadata = dict(
            zip(list(data["metadata_keys"]), list(data["metadata_values"]), strict=True)
        )
        return cls(
            origin_m=data["origin_m"],
            resolution_m=float(data["resolution_m"]),
            shape=tuple(data["count"].shape),
            count=data["count"],
            pitch_min=data["pitch_min"],
            pitch_max=data["pitch_max"],
            roll_bins=data["roll_bins"],
            yaw_sin=data["yaw_sin"],
            yaw_cos=data["yaw_cos"],
            sigma_min_best=data["sigma_min_best"],
            limit_margin_best=data["limit_margin_best"],
            metadata=metadata,
        )

    # ---------------------------------------------------------------- queries

    @property
    def coverage(self) -> float:
        """Fraction of voxels ever reached."""
        return float(self._occupied.mean())

    def yaw_at(self, index: tuple[int, int, int]) -> float:
        """Circular mean of the TCP yaw observed in one voxel, radians."""
        return float(np.arctan2(self.yaw_sin[index], self.yaw_cos[index]))

    def index_of(self, position: np.ndarray) -> tuple[int, int, int] | None:
        """Voxel containing a position, or None if it is outside the grid."""
        index = np.floor((np.asarray(position, float) - self.origin_m) / self.resolution_m)
        index = index.astype(np.int64)
        if np.any(index < 0) or np.any(index >= np.array(self.shape)):
            return None
        return tuple(int(v) for v in index)

    def nearest_occupied(self, position: np.ndarray) -> tuple[int, int, int]:
        """Populated voxel closest to a position. Never fails: the grid is not empty."""
        distances = np.linalg.norm(self._occupied_centres - np.asarray(position, float), axis=1)
        return tuple(int(v) for v in self._occupied_index[int(np.argmin(distances))])

    def roll_is_feasible(self, index: tuple[int, int, int], roll: float) -> bool:
        bin_index = int(np.floor((roll + np.pi) / (2 * np.pi) * ROLL_BINS)) % ROLL_BINS
        return bool(int(self.roll_bins[index]) & (1 << bin_index))

    def project(self, pose: SO100TaskPose) -> AtlasProjection:
        """Pull a requested task pose onto something the arm was seen to hold.

        Position is not moved unless it falls outside the sampled set entirely;
        the box clamp upstream is responsible for position, and moving it here
        as well would give the operator two walls at different places. Pitch is
        clamped into the voxel's measured interval. Roll is checked but not
        changed -- an unoccupied roll arc more often means the atlas is
        undersampled there than that the arm cannot do it, and silently rotating
        the gripper is worse than letting the IK report the error.
        """
        index = self.index_of(pose.position)
        outside = index is None or self.count[index] == 0
        if outside:
            index = self.nearest_occupied(pose.position)

        low = float(self.pitch_min[index])
        high = float(self.pitch_max[index])
        pitch = float(np.clip(pose.pitch, low, high))

        return AtlasProjection(
            pose=pose.replace(pitch=pitch),
            pitch_clamped=not np.isclose(pitch, pose.pitch),
            roll_infeasible=not self.roll_is_feasible(index, pose.roll),
            voxel_count=int(self.count[index]),
            outside_atlas=bool(outside),
        )

    def pitch_envelope(self, min_count: int = 1) -> tuple[float, float]:
        """Widest pitch reached at *any* sufficiently sampled voxel.

        This is what `TaskLimits.pitch_min_rad` / `pitch_max_rad` should hold.
        The coarse clamp is an envelope, not a guarantee: it must not forbid
        anything the atlas would allow anywhere, because the per-voxel interval
        behind it is what actually constrains the target.

        The intersection over all voxels -- the interval available *everywhere*
        -- would be the wrong choice and is nearly empty in any case: at the
        edge of reach the arm is almost straight and pitch is pinned to within a
        degree or two. See :meth:`pitch_universal`.
        """
        usable = (self.count >= min_count) & np.isfinite(self.pitch_min)
        if not usable.any():
            raise ValueError(f"no voxel has {min_count} samples; build a larger atlas")
        return float(self.pitch_min[usable].min()), float(self.pitch_max[usable].max())

    def pitch_universal(self, min_count: int = 1) -> tuple[float, float] | None:
        """Pitch interval available at *every* sufficiently sampled voxel.

        None when there is no such interval, which is the usual answer and the
        reason the coarse clamp is an envelope instead.
        """
        usable = (self.count >= min_count) & np.isfinite(self.pitch_min)
        if not usable.any():
            return None
        low = float(self.pitch_min[usable].max())
        high = float(self.pitch_max[usable].min())
        return (low, high) if low < high else None
