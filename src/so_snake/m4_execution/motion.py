"""Joint-space point-to-point motion, shared by the real-arm scripts.

A plain, IK-free approach: each step the arm is commanded a small, rate-limited
step from where it currently is toward the target joints, so it walks there
monotonically. No task-space target and no IK means no configuration flips -- the
right primitive for "go to this known pose" (homing, move-to-start) as opposed to
teleoperation.

**A move that did not arrive is not one fact but three**, and the caller has to
tell them apart: the arm is still creeping in and ran out of iterations; the arm
settled a degree or two out and physically cannot do better; something is
holding it. The first two are normal and the third is not, so this returns a
`MoveOutcome` -- which is truthy when the move arrived, so old `if reached:`
callers are unchanged -- carrying the residual per joint and whether progress
had stopped. `EpisodeReplayer` uses exactly that to decide whether to play a
recording back, and the alternative (a bare False) is what made a real-arm
replay abort and drop torque because the shoulder was 2.7 degrees out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .backends import RobotBackend

# How long the residual has to stop improving before the move is called stalled,
# and by how little. One second at 30 Hz: shorter mistakes the servo's own
# settling time for a stall, and 0.15 deg is under the encoder's own resolution
# per step but well over its noise, so genuine slow creep still counts as motion.
STALL_WINDOW_S = 1.0
STALL_PROGRESS_DEG = 0.15


@dataclass(frozen=True)
class MoveOutcome:
    """How a point-to-point move ended.

    Truthy when the arm arrived, so `if move_to_joints(...):` still reads the
    way it always did.
    """

    reached: bool
    residual_deg: np.ndarray  # |target - measured| per joint, at the end
    joint_names: tuple[str, ...] = ()
    stalled: bool = False  # progress stopped before the tolerance was met
    interrupted: bool = False  # `should_continue` said stop
    steps: int = 0

    def __bool__(self) -> bool:
        return self.reached

    @property
    def max_residual_deg(self) -> float:
        return float(np.max(self.residual_deg)) if len(self.residual_deg) else 0.0

    def worst(self, limit: int = 3) -> list[tuple[str, float]]:
        """The joints furthest from the target, worst first."""
        names = self.joint_names or tuple(f"j{i}" for i in range(len(self.residual_deg)))
        order = np.argsort(-np.asarray(self.residual_deg))
        return [(names[i], float(self.residual_deg[i])) for i in order[:limit]]

    def describe(self, limit: int = 3) -> str:
        """"shoulder_pan 2.7 deg, wrist_flex 1.1 deg" -- for a log line."""
        return ", ".join(f"{name} {value:.1f} deg" for name, value in self.worst(limit))


def move_to_joints(
    backend: RobotBackend,
    target_deg: np.ndarray,
    *,
    step_deg: float = 6.0,
    tol_deg: float = 3.0,
    hz: float = 30.0,
    max_extra_steps: int = 200,
    on_progress: Callable[[float], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> MoveOutcome:
    """Walk `backend` to `target_deg` (in the backend's joint frame).

    `step_deg` is how far the commanded goal leads the measured position each step.
    It doubles as the servo's drive: a Feetech in position mode applies torque
    roughly proportional to (goal - present), so too small a lead starves it and
    the arm stalls under gravity (this is why 1.5 deg could not lift the elbow,
    while teleop -- which leads by up to max_joint_step ~6 deg -- moved fine). It
    is bounded above by the backend's own `max_relative_target`, so keep that >=
    `step_deg` or the hardware clamp will re-starve the drive.

    `tol_deg` defaults to `TeleopConfig.joint_settle_tol_deg`'s value, which is
    the servo's *measured* standing offset on this bench and not a preference --
    see the note there. Asking for better than the servo can hold means every
    move reports failure.

    The move ends when the arm arrives, when progress stops (`stalled`: the
    residual has not improved for `STALL_WINDOW_S`, which is an obstruction, a
    joint limit, or a lead too small to overcome gravity), when the iteration
    budget runs out, or when `should_continue` says stop. All four are reported
    rather than collapsed into one False, because only the first is normal and
    only the third and fourth mean *do not carry on*.

    `should_continue` is polled between steps, like the teleoperation loop's. A
    move that cannot be interrupted is one the operator's stop button does not
    reach. Ending the move leaves the arm holding where it is; it is not released.
    """
    if step_deg <= 0 or tol_deg <= 0 or hz <= 0:
        raise ValueError("step_deg, tol_deg and hz must be positive")
    target = np.asarray(target_deg, float)
    measured = np.asarray(backend.read_joints_deg(), float)
    if target.shape != measured.shape:
        raise ValueError(f"target has {target.shape} joints, backend reports {measured.shape}")

    names = tuple(getattr(backend, "joint_names", ()) or ())
    stall_steps = max(2, int(round(STALL_WINDOW_S * hz)))
    best = float("inf")
    since_progress = 0

    def outcome(reached: bool, err: np.ndarray, steps: int, **flags: bool) -> MoveOutcome:
        return MoveOutcome(
            reached=reached,
            residual_deg=np.abs(np.asarray(err, float)),
            joint_names=names,
            steps=steps,
            **flags,
        )

    max_iters = int(np.abs(target - measured).max() / step_deg) + max_extra_steps
    err = target - measured
    for step in range(max_iters):
        if should_continue is not None and not should_continue():
            return outcome(False, err, step, interrupted=True)
        measured = np.asarray(backend.read_joints_deg(), float)
        err = target - measured
        worst = float(np.abs(err).max())
        if worst <= tol_deg:
            return outcome(True, err, step)

        # Progress is measured against the best the move has ever done, not
        # against the previous step: a servo hunting around a standing offset
        # improves and un-improves every step without getting anywhere.
        if worst < best - STALL_PROGRESS_DEG:
            best, since_progress = worst, 0
        else:
            since_progress += 1
            if since_progress >= stall_steps:
                return outcome(False, err, step, stalled=True)

        backend.write_joints_deg(measured + np.clip(err, -step_deg, step_deg))
        if on_progress is not None:
            on_progress(worst)
        sleep(1.0 / hz)
    return outcome(False, err, max_iters)
