# Plan: migrate to the SO-100's real five-dimensional task space

**Status: implemented for the offline Phase 0 path.** The control path now uses
`SO100TaskPose(x, y, z, pitch, roll)`, `TaskFrame`, `OrientationProjector`,
`TaskIK5D`, `ClutchRetargeter`, the feasibility atlas, and the mock/MuJoCo-safe
joint boundary. Hardware Gate C and replay Gate D are still open.

Continues [`five_dof_orientation.md`](five_dof_orientation.md), which established
that the arm is 5-DoF and the action space must say so. This document settles
*how*.

## The frozen contract

Nothing below is to be changed without re-recording data, because the recorded
dataset's action space depends on all of it.

| layer | contract |
|---|---|
| device input | absolute IMU quaternion + clutch state, raw sticks, gripper |
| teleop mapping | clutch-relative orientation, anchored on measured joints |
| orientation control | position-anchored chart projection onto `span{a(p), u(p,pitch,roll)}`; `J_w · null(J_v)` retained as an independent diagnostic |
| internal target | absolute `(x, y, z, pitch, roll)` |
| policy action | `Δ(x, y, z, pitch, roll)` + gripper |
| IK | true 5D task IK — no yaw objective, no orientation weight to tune |
| logging | full 6D FK pose + raw IMU + projected action + executed joints |

## Evidence

### The controllable orientation space is exactly two-dimensional

With TCP position held fixed, achievable angular velocities span

```
N = null(J_v)   in R^{5x2}      null space of the position task
B = J_w · N     in R^{3x2}      achievable angular directions there
```

Measured over 12 configurations sampled inside the workspace: `rank(B) = 2` in
every one, with the third singular value exactly 0.0000. The arm has no third
independent orientation degree of freedom.

Consistent with the earlier fixed-position solve (600 seeds, 230 hits): TCP roll
spans ±180 deg, pitch spans [−67.3, +62.8] deg, and `shoulder_pan` moves only
±2.45 deg (std 0.81), so TCP yaw follows position rather than being commanded.

### "Strip the Euler yaw" would be actively wrong

The uncontrollable direction — the normal to the plane spanned by `B` — is not a
fixed axis. Across those 12 configurations its mean is `[0.300, 0.011, 0.853]`
with standard deviation `[0.345, 0.165, 0.190]`, and its angle from world +Z
averages 24.96 deg, reaching roughly 65 deg. One sampled configuration had it at
`[0.9075, −0.0582, 0.4160]` — predominantly world **X**.

Removing the world-Z component there deletes a *controllable* direction while
leaving an *uncontrollable* one in the command. The projection must be computed
from the current configuration, not from a named axis.

### The two controllable directions have unequal gain

Singular values of `B` are `sigma_1 ~ 0.0175` and `sigma_2 ~ 0.006–0.009`, a
ratio of 2–3x. The same commanded rotation magnitude produces two to three times
more joint motion in one direction than the other. The DLS damping has to
account for this or the operator will feel the arm respond unevenly depending on
which way they twist.

### The achievable set is affine, not linear

For a commanded linear velocity `v*`, achievable angular velocities are

```
omega  in  omega_0(v*) + span(B)
```

`omega_0` is the "yaw follows position" coupling, and it is not small. Minimum-
norm joint solutions for 1 mm of pure translation force:

| axis | median | p95 | max |
|---|---|---|---|
| x | 0.148 deg | 0.241 deg | 0.289 deg |
| y | 0.201 deg | 0.307 deg | 0.371 deg |
| z | 0.237 deg | 0.281 deg | 0.331 deg |

At 30 Hz with a 1.5 mm translation step that is about **13.8 deg/s** of
orientation dragged along by translation alone — larger than the ~7 deg/s of
deliberate rotation input the current loop is tuned for.

**Consequence:** projecting the IMU delta onto `span(B)` is correct only when
the operator is not translating. The position-dependent base orientation is
therefore not an optional refinement:

```
R_target = R_position_yaw(p) · R_pitch(theta_p) · R_roll(theta_r) · R_tool_offset
```

Projection handles the increment; the position-dependent base handles the offset.
Implementing either alone leaves a drift.

### Writing our own 5D IK also removes three measured placo problems

Independently of the 5-DoF argument, replacing placo's frame task with a 5x5
damped least squares solve fixes:

1. **Solver statefulness.** placo's solver carries state between calls, so
   solving the same target twice gives different answers depending on history.
   This silently invalidated several of our measurements before it was noticed.
2. **Local minima.** On a 14^3 workspace grid, 17 points looked unreachable, the
   worst missing by 360.3 mm; every one solved to 0.0000 mm from a better seed.
   The current workaround is multi-seed retry, which in closed loop risks
   jumping IK branches and had to be guarded against.
3. **Orientation weight tuning.** `ik_orientation_weight` exists only because a
   6-DoF target has to be compromised. With no yaw objective there is nothing to
   weight.

An incremental teleoperation loop always has a good seed, so a local solver is
the right tool — it just needs to be one we control.

## What this replaces

The current code reached these numbers by patching a 6-DoF formulation, and all
of it becomes unnecessary:

| current mechanism | why it exists | after |
|---|---|---|
| `ik_orientation_weight = 0.001` | 6-DoF target must be compromised | deleted |
| `orientation_feasibility_feedback` | orientation target ran away | demoted to diagnostics |
| `rotation_step_rad = 0.004` (~7 deg/s) | slowing down the runaway | retune freely |
| `ik_retry_tolerance_m` / `ik_max_retries` / `ik_max_retry_jump_deg` | placo local minima | deleted |
| `WorkspaceConfig.rpy_min/max_rad` | never a valid model of reachability | deleted |

The position box stays. It is measured, not inherited:

```
x [0.170, 0.360]   y [-0.200, 0.200]   z [0.080, 0.200]     100% reachable at 14^3
```

For the avoidance of doubt, joycon-robotics' original box — `x [0.125, 0.380]`,
`y [-0.4, 0.4]`, `z [0.046, 0.23]` — is **not** to be used as a coarse filter.
It measured 84% reachable, and its `|y| <= 0.4 m` is geometrically impossible:
the TCP sweeps a circle of radius 0.310 m about the shoulder_pan axis.

## Order of work

Foundations first. Steps 1 and 2 decide everything above them; once they are
measured and settled, the rest is wiring.

### Step 1 — `OrientationProjector`

`src/so_snake/m3_safety/projection.py` — implemented.

The online projection uses the position-anchored chart from `TaskFrame`: at fixed
position the controllable angular directions are the orthonormal pair `a` (plane
normal, pitch axis) and `u` (tool approach axis), so resolving an IMU rotation is
the closed-form projection `d_pitch = -a·omega`, `d_roll = u·omega`.

`J_w · null(J_v)` remains in the module as the independent check. In workspace
samples the chart/Jacobian controllable planes are gated by
`tests/test_kinematics.py`.

### Step 2 — 5D task IK

`src/so_snake/m3_safety/ik5d.py`

```
e(q)   = [ p* - p(q) ,  e_pitch(q) ,  e_roll(q) ]        in R^5
J_task = [ J_v ; b_pitch^T J_w ; b_roll^T J_w ]          in R^{5x5}
dq     = J_task^T (J_task J_task^T + lambda^2 I)^{-1} e
```

with `lambda` scheduled on the smallest singular value so it damps near
singularities rather than being a fixed constant. Keep placo for FK and as an
independent cross-check on the new solver, not in the control path.

Gate: near-seed and cold-seed solves are covered by `tests/test_kinematics.py`,
with samples restricted to the measured workspace and away from the mechanical
stops, matching the measured failure mode: bad cold starts concentrate within a
few degrees of joint limits.

### Step 3 — 5D target state

`src/so_snake/m3_safety/task_pose.py`, replacing `EETargetTracker` — implemented.

- `SO100TaskPose(x, y, z, pitch, roll)`
- `R_position_yaw(p) = atan2(y, x - 0.0452)` about the measured pan axis
- atlas validation closes the former open item: voxel mean yaw vs closed form
  has median error 0.09 deg and max 1.02 deg over populated workspace cells

### Step 4 — `ClutchRetargeter`

`src/so_snake/teleop/clutch.py`

- rising edge latches `imu_ref` and a task pose taken from **measured joint FK**,
  never from the last commanded target — otherwise re-clutching while a tracking
  error is present produces a jump
- held: `omega = log(R_imu_ref^T · R_imu_now)`, projected, added to the latched
  pitch/roll; sticks integrate position
- released: target frozen, operator free to reposition their hands

### Step 5 — device contract

`NintendoProSample`: timestamp, both sticks, IMU quaternion, clutch, gripper.
Raw and robot-agnostic. The device layer does not know how many orientation
degrees of freedom the robot has, which are controllable, or where the TCP is,
so it must not be the layer that decides.

### Step 6 — feasibility atlas

`scripts/build_feasibility_atlas.py` — implemented; current checked-in atlas was
built from 40 M samples with 1 deg joint-limit margin.

Sample joint space, FK, bin by XYZ voxel, and record per voxel: reachable pitch
interval, roll occupancy bins, yaw agreement, manipulability, distance to joint
limits.

Online: `5D target -> XYZ box clamp -> atlas nearest-cell projection -> 5D IK ->
joint safety`.

### Step 7 — diagnostics

Rename `orientation_feasibility_feedback` to `orientation_projection_feedback`
and demote it out of the control loop, but keep it logging:

```
projected_pitch_delta      ik_pitch_error
projected_roll_delta       ik_roll_error
rejected_rotation_norm     orientation_saturated
yaw_residual_for_diagnostics
```

A large `rejected_rotation_norm` means the operator is pushing in a direction
the arm does not have — which is also a useful feasibility label for the
active-perception policy later.

## Gates before recording data

| gate | needs hardware | checks |
|---|---|---|
| **A** projection | no (synthetic IMU), re-run with controller later | pure controller yaw yields ~0 pitch/roll; pure pitch yields mostly pitch; twist about the controller's forward axis yields mostly roll; no discontinuity at large angles; re-clutch causes no jump |
| **B** 5D IK | no | success >= 99.5% on atlas-drawn targets; XYZ error within grasp tolerance; stable pitch/roll error; no branch jumping |
| **C** closed loop | yes | 30 s clutch held with no target drift; release, reposition, re-clutch with no jump; pure wrist yaw does not move the arm; smooth saturation at the pitch boundary; IK failure does not accumulate target into illegal regions |
| **D** replay | no | replay `task action 5D + gripper` through the same projector and IK; trajectory close to the recorded joint commands; no ambiguous action dimensions; no legacy yaw fields; stable normalisation statistics |

Gate A/B are now covered by the pytest suite. Gate C still needs hardware. Gate D
(replay) is still open.

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` currently passes 34 tests. The
default offline loop gate is `scripts/check_teleop_loop.py --steps 300` and
passes with position p95 0.0015 mm, pitch p95 0.0056 deg, solver convergence
98.3%, and max command step 1.38 deg. The deliberate stress case
`scripts/check_teleop_loop.py --steps 300 --rotation-amplitude 0.35` reproduces
the former failure: pitch p95 6.68 deg, atlas pitch clamp 25.3%, solver
convergence 57.7%, and max position error 99.8 mm.

## Blueprint amendments

**M2 — VLA Policy.** Action head predicts `Δ(x, y, z, pitch, roll)` + gripper,
not `Δq` and not a 6-DoF delta. Auxiliary heads may predict phase
(VIEW/MANIPULATE/VERIFY), observation intent, the current 5D target, and
orientation saturation.

**M3 — feasibility projection & safety.** Becomes:

```
Δtask5 + gripper
  -> 5D target integrator
  -> XYZ coarse clamp
  -> position-conditioned pitch/roll projection
  -> 5D constrained IK
  -> joint limits / Δq cap / collision
```

**M4 — execution.** Accepts safe joint commands only. No 6-DoF EE pose crosses
this boundary.

## Dataset layout

Three action streams, so that changing the projector or the IK later does not
require re-recording:

```
action.raw.sticks              action.task.delta_{x,y,z}
action.raw.imu_quaternion      action.task.delta_{pitch,roll}
action.raw.clutch              action.task.gripper

action.joint.{shoulder_pan, shoulder_lift, elbow_flex,
              wrist_flex, wrist_roll, gripper}
```

`action.task.*` is the policy's training target. The other two make the chain
*raw intent -> projected intent -> executed joints* auditable offline.

Observations keep the full 6-DoF pose — a 5D action space does not imply a 5D
observation, and the complete FK pose stays necessary for diagnostics,
multi-view geometry, and camera extrinsics:

```
observation.state.joints          observation.state.ee_position
observation.state.task_pose_5d    observation.state.ee_quaternion
                                  observation.state.ee_rotvec
```

## Open items

- `R_position_yaw(p)` may not have a clean closed form across IK branches.
  Sample and validate before committing to one; leaving it implicit in the
  solver is an acceptable first version.
- The atlas needs re-running after any hardware calibration change, since
  calibration moves the reachable set.
- Roll is periodic in principle but constrained in practice by motor range,
  cabling, the wrist camera's cable, self-collision, and the TCP mount. None of
  those are in the URDF, so the atlas will overestimate roll freedom until
  measured on hardware.
- Everything here is derived from the URDF. No number in this document has been
  checked against the physical arm.
