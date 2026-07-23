# The SO-100 has five degrees of freedom, and the action space must say so

**Implemented decision.** This document records the measurement that forced the
migration. The current control path is implemented in the 5D task space described
in `plan_5dof_task_space.md`.

## What we measured

The SO-100's arm chain is 5-DoF (`shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex`, `wrist_roll`) plus a gripper. A full TCP pose is six constraints.
One of them therefore cannot be commanded — the question is which.

Solving position-only IK to a fixed point from 600 random seeds, 230 of them
reach it to within 1 mm. Reading back what those solutions achieve:

| quantity | range across solutions | std |
|---|---|---|
| TCP roll | −180.0 … +180.0 deg | 163.8 |
| TCP pitch | −67.3 … +62.8 deg | 32.6 |
| TCP yaw | −162.9 … +164.3 deg | 51.4 |
| `shoulder_pan` | **−2.45 … +2.45 deg** | **0.81** |
| `wrist_roll` | −180.0 … +180.0 deg | 154.0 |

`shoulder_pan` barely moves. Fixing the TCP position fixes the pan angle, and
the pan angle is what sets which way the gripper faces. **TCP yaw is determined
by position; it is not an axis the operator can command.** What remains free is
`wrist_roll` (gripper roll, fully free) and the shoulder/elbow/wrist redundancy
in the vertical plane (gripper pitch, roughly ±65 deg).

So the arm's real action space is:

```
position (x, y, z)  +  gripper pitch  +  gripper roll  =  5
```

## Why this broke things in practice

Treating the target as a 6-DoF pose caused every anomaly we chased:

- The configured home orientation, identity, turned out to be **144.6 deg away**
  from the nearest orientation achievable at the home position. The arm could
  never settle at home.
- The inherited roll/pitch/yaw box from joycon-robotics excluded the home pose's
  own orientation of (180, 40, 180) deg.
- Letting the operator integrate roll, pitch and yaw independently walked the
  target steadily into unreachable orientations. The solver then spent itself
  chasing them and dragged position along: position p95 degraded from 0.66 mm
  at 300 steps to 46 mm at 600 purely because the target had longer to run away.

## What replaced the stopgap

The old `orientation_feasibility_feedback` stopgap is gone. The target is now
`SO100TaskPose(x, y, z, pitch, roll)`: yaw is derived from position about the
measured pan axis, orientation input is projected into pitch/roll, and the
feasibility atlas clamps pitch by position before `TaskIK5D` runs.

The default offline loop gate, `scripts/check_teleop_loop.py --steps 300`, now
passes with position p95 0.0015 mm, pitch p95 0.0056 deg, solver convergence
98.3%, and max command step 1.38 deg. The old runaway can still be reproduced as
a stress test with `--rotation-amplitude 0.35`, which intentionally drives into
atlas and joint-rate walls.

## Recommendation

This recommendation has been accepted and implemented for the offline Phase 0
path: reparameterise the target as **position + gripper pitch + gripper roll**,
and drop yaw from the operator's control entirely, deriving it from position.

> **Settled.** This was accepted, with one correction: the uncontrollable
> direction is not world yaw and must not be removed by name. It varies with
> configuration, reaching ~65 deg from world +Z, so it has to be computed from
> the Jacobian at each step. The frozen contract and the implementation plan are
> in [`plan_5dof_task_space.md`](plan_5dof_task_space.md).

Consequences worth weighing before doing it:

- The excursions above should disappear, because an infeasible orientation can
  no longer be expressed.
- The Pro controller mapping gets simpler and more honest: two sticks for
  position, one trigger axis for pitch, one for roll.
- **It changes M2's action space too.** The VLA's action head would predict
  `Δ(x, y, z, pitch, roll)` plus gripper rather than a 6-DoF delta. Better to
  settle this before recording the dataset that trains it, since re-recording
  costs far more than deciding now.

The rejected alternative was to keep a 6-DoF target and rely on feasibility
feedback. That would have kept a permanent gap between what the operator asks
for and what the arm does, showing up in the recorded data as actions that were
never actually executed.
