"""Joint-space point-to-point motion, shared by the real-arm scripts.

A plain, IK-free approach: each step the arm is commanded a small, rate-limited
step from where it currently is toward the target joints, so it walks there
monotonically. No task-space target and no IK means no configuration flips -- the
right primitive for "go to this known pose" (homing, move-to-start) as opposed to
teleoperation.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .backends import RobotBackend


def move_to_joints(
    backend: RobotBackend,
    target_deg: np.ndarray,
    *,
    step_deg: float = 6.0,
    tol_deg: float = 1.0,
    hz: float = 30.0,
    max_extra_steps: int = 200,
    on_progress: Callable[[float], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Walk `backend` to `target_deg` (in the backend's joint frame). Returns reached.

    `step_deg` is how far the commanded goal leads the measured position each step.
    It doubles as the servo's drive: a Feetech in position mode applies torque
    roughly proportional to (goal - present), so too small a lead starves it and
    the arm stalls under gravity (this is why 1.5 deg could not lift the elbow,
    while teleop -- which leads by up to max_joint_step ~6 deg -- moved fine). It
    is bounded above by the backend's own `max_relative_target`, so keep that >=
    `step_deg` or the hardware clamp will re-starve the drive. Iterations are
    bounded so an obstruction or joint limit ends the move (returning False)
    instead of looping forever.

    `should_continue` is polled between steps, like the teleoperation loop's. A
    move that cannot be interrupted is one the operator's stop button does not
    reach: without this, "stop" during a homing move means waiting for the arm
    to finish walking there first. Returning False ends the move where it is and
    reports False -- the arm holds that position, it is not released.
    """
    if step_deg <= 0 or tol_deg <= 0 or hz <= 0:
        raise ValueError("step_deg, tol_deg and hz must be positive")
    target = np.asarray(target_deg, float)
    measured = np.asarray(backend.read_joints_deg(), float)
    if target.shape != measured.shape:
        raise ValueError(f"target has {target.shape} joints, backend reports {measured.shape}")

    max_iters = int(np.abs(target - measured).max() / step_deg) + max_extra_steps
    for _ in range(max_iters):
        if should_continue is not None and not should_continue():
            return False
        measured = np.asarray(backend.read_joints_deg(), float)
        err = target - measured
        if np.all(np.abs(err) <= tol_deg):
            return True
        backend.write_joints_deg(measured + np.clip(err, -step_deg, step_deg))
        if on_progress is not None:
            on_progress(float(np.abs(err).max()))
        sleep(1.0 / hz)
    return False
