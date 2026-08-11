# Real SO-100 bring-up: decisions and gotchas

How the offline control stack was connected to the physical SO-100, and the
non-obvious things the hardware taught us. Companion to the README's "真机
preflight / 关节坐标映射 / 真机遥操" sections.

## Dependency: the lerobot fork, not upstream

The Switch Pro `NintendoTeleop` and the `SOFollower` API this project targets live
in the lab fork **`linkage-x/lerobot`, branch `box`** — not in
`huggingface/lerobot` main (which has only a generic `gamepad` teleop and a
`SOFollowerConfig` that is missing `id`/`calibration_dir`). The `teleop` extra
pins that branch. Install needs SSH access and `GIT_LFS_SKIP_SMUDGE=1` (a lerobot
LFS test artifact is missing on the remote and otherwise fails the whole install).

`SOFollowerBackend` uses `SOFollowerRobotConfig` (= `RobotConfig` + `SOFollowerConfig`),
not the bare `SOFollowerConfig`; the bare one lacks `id`, so lerobot's base
`Robot.__init__` (which reads `config.id`) rejects it.

## Finding the hardware

Nothing takes a mandatory `--port` any more: `so_snake.devices` identifies the
driver board by its USB bridge chip (`1a86:55d3`, a WCH CH343, on this bench),
excludes the ports that are never an arm (macOS Bluetooth, debug console), and
resolves a single candidate **without opening anything** — so detection cannot
disturb a running session. Two USB serial adapters attached, and it falls back to
a read-only servo ping and takes whichever answers; still ambiguous, and it
refuses with the list rather than picking. `--port` and `SO_SNAKE_ARM_PORT`
override it, in that order. `scan_devices.py` shows all of this.

Cameras are the opposite case and stay manual on purpose: an OpenCV index cannot
be mapped to a device name on macOS (see `so_snake.m0_perception`), so
`scan_devices.py` writes a thumbnail per index and roles are assigned by eye.
`--camera <role>=auto` exists but only resolves when exactly one unclaimed camera
is attached — with two it refuses, because a wrist-labelled third-person view is
not detected by anything downstream.

## The servo settles a couple of degrees out, and that is normal

Measured from the recorded takes (`observation.state.joints_deg` minus
`action.joint.commanded_deg`, over frames where the arm was standing still at
the start pose):

    shoulder_pan  2.7    shoulder_lift  1.2    elbow_flex  0.8
    wrist_flex    1.1    wrist_roll     0.9        (degrees, mean)

The STS3215 in position mode is a proportional controller and lerobot's
`configure()` halves its gain (`P_Coefficient` 32 -> 16, "to avoid shakiness"),
so every loaded joint holds with a standing offset. Nothing is wrong with the
arm; it simply cannot be commanded to a degree.

This is why `TeleopConfig.joint_settle_tol_deg` is 3.0 and not 1.0. With 1.0,
`move_to_joints` never reported success, and `EpisodeReplayer` treated that as
fatal -- so a real-arm replay walked the arm to the episode's first pose,
aborted, and released torque without playing a frame. "Arrived" and "stuck" are
now separate questions: `joint_stuck_deg` (8.0) is the one that stops a move,
and it reports which joints and by how much.

## Bring-up order

0. `scan_devices.py` — what is attached (optional; the port is detected anyway).
1. `preflight_real_arm.py [--probe] [--scan-cameras]` — deps, joint contract,
   port, controller, cameras, calibration file; `--probe` pings the servos
   read-only (no torque, no motion).
2. `lerobot-calibrate --robot.type=so100_follower --robot.port=<PORT> --robot.id=so_snake`
   — interactive, moves the arm; writes `~/.cache/huggingface/lerobot/.../so_snake.json`.
3. `map_joint_frames.py draft / signs / check` — build + verify the lerobot↔URDF map.
4. `move_to_start.py` — go to a known start pose inside the workspace.
5. `teleop_real_arm.py` — teleoperate (it also does 4 on startup).

## The lerobot ↔ URDF joint-frame map

so-snake's kinematics/IK/safety are all in the **official SO-ARM100 URDF** frame.
lerobot's `SOFollower` reads/accepts **degrees in its own calibration frame**,
whose zero is the *midpoint of the range you swept during calibration* and whose
direction is set by how each servo is mounted. The two differ, per joint, by an
affine map:

    q_urdf = sign * q_lerobot + offset       sign ∈ {+1, −1}     (arm joints)

This is exact, with **unit gain**: lerobot `DEGREES` mode is `(raw − mid)·360/4095`
with `mid = (range_min+range_max)/2`, so both frames are linear in the raw encoder
tick with the same |slope|. Consequences that drove the tool design:

- **offset is recoverable with no motion.** The recorded range is symmetric about
  lerobot 0, so `offset ≈ (urdf_lower + urdf_upper)/2` = the URDF midpoint.
  `map_joint_frames.py draft` computes this straight from the calibration file.
- **sign is NOT recoverable from data.** It depends on servo mounting; and because
  the range is symmetric about 0, flipping the sign yields the *same* set of swept
  values — so a range sweep cannot reveal it. It needs one physical observation.
  `signs` gets it by having the operator drive each joint to its two **hard stops**
  and answer one question about that joint's **own driven link** (upper-arm
  front/back, forearm straight/folded, tool up/down) — never the gripper's world
  height, which runs through the whole chain and is even non-monotonic in
  shoulder_lift. The correct answer per joint is precomputed from per-link FK.
- **the map is an exact bijection**, so a value read and written straight back
  round-trips to itself. That is what keeps the arm from jumping on the first
  command even when `offset` is only approximate.

`SOFollowerBackend(joint_map=…)` applies it: reads map lerobot→URDF, writes URDF→lerobot.

Known limitation: `offset = URDF midpoint` is only right if calibration swept each
joint symmetrically to its true mechanical stops. Ours under-swept shoulder_pan and
wrist_flex, and the elbow's physical range exceeds the URDF software limit, so those
offsets carry a few degrees of error — fine for cautious teleop, worth refining.

## Servo drive: why move-to-start stalled

lerobot's `configure()` sets the arm servos' `P_Coefficient` to **16** (half the
STS3215 default, to reduce shakiness). A Feetech in position mode applies torque
roughly proportional to `(goal − present)`. The first `move_to_start` commanded the
goal only **1.5°** ahead of measured, so drive torque ∝ 16·1.5 — too weak to lift
the elbow against gravity; the arm stalled and had to be dragged by hand. Teleop
did not stall because its IK command leads by up to `max_joint_step ≈ 6°`.

Fix: `move_to_joints` leads the goal by `step_deg` (default **6°**, matching
teleop's drive), and `max_relative_target` must be ≥ `step_deg` or the hardware
clamp re-starves it. If 6° is still not enough, the next lever is raising
`P_Coefficient` back toward 32 (trades some shakiness for holding torque).

## Clutch / loop fixes found on hardware

- **Rotation sign.** The controller IMU's rotation sense was flipped relative to
  the world convention the projector expects, so wrist tilt turned the tool the
  wrong way. The projector is linear in `omega`, so a single negation in
  `ClutchRetargeter` flips both pitch and roll consistently (logs included).
- **Post-release "motion tail".** The task target integrates open-loop while the
  clutch is held, leading the physical arm by the servo lag. Three integrators
  hold that lead — the retargeter target, the tracker pose, and the IK seed. On
  release we snap the **retargeter target** (ClutchRetargeter) and the **tracker
  pose** (TeleopLoop) to the measured pose. We deliberately do **not** touch the
  IK seed (`last_command`): seeding IK from the measurement lets the solve drift
  between IK branches, which showed up as the arm jumping to a different
  configuration when the clutch was pressed again.

## Gripper torque on disconnect

lerobot's `disconnect(disable_torque=True)` clears the serial port *before*
issuing the per-motor disable writes, which can leave the last motor on the bus —
the gripper — energized. `SOFollowerBackend.disconnect` disables torque explicitly
on a fully-open port first, so the gripper relaxes when the program exits.
