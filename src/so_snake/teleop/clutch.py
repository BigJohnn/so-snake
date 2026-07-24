"""The clutch retargeter (plan step 4): raw controller frames to a 5D target.

This is the layer that knows what the arm is. The device reports sticks and an
attitude; this decides that the arm has three translational and two rotational
coordinates and maps one onto the other.

## The clutch

  * **rising edge** — latch the controller's attitude as a reference, and latch
    the arm's task pose taken from **measured joint forward kinematics**, never
    from the last commanded target. A tracking error is always present, and
    latching the command means the operator's next gesture starts from a pose
    the arm is not actually in; the error then shows up as a jump the moment
    they move.
  * **held** — the attitude *difference* since the reference is projected onto
    pitch and roll and added to the latched values, and the sticks integrate
    position. Orientation is absolute-referenced rather than integrated, so a
    dropped frame costs nothing and gyro drift cannot accumulate.
  * **released** — the target freezes. The operator can put their hands back
    where they want them, and nothing moves until they clutch in again.

## Why the attitude difference is composed in the world frame

`R_delta = R_now . R_ref^T`, not `R_ref^T . R_now`. The two differ by which
frame the difference lives in, and the operator is watching a fixed-base arm
across a table: "tilt down" has to mean down in the room. Body-frame
composition inverts that whenever the gripper is rolled — which it is at the
home pose, whose roll is 90 degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import TeleopConfig
from ..kinematics import quaternion_to_matrix, rotation_log
from ..m3_safety.projection import OrientationProjector
from ..m3_safety.task_pose import SO100TaskPose, wrap_to_pi
from .sources import NintendoProSample


@dataclass
class RetargetResult:
    """One retargeting step: the target it produced, and why."""

    target: SO100TaskPose
    engaged: bool  # was the clutch held this step
    latched: bool  # did this step re-latch (clutch rising edge)
    delta_position_m: np.ndarray  # (3,) what the sticks asked for, in metres
    projected_pitch_delta: float  # rad, relative to the latched pitch
    projected_roll_delta: float  # rad, relative to the latched roll
    rejected_rotation_norm: float  # rad, the part of the gesture the arm cannot do


@dataclass
class ClutchRetargeter:
    """Turns `NintendoProSample`s into an absolute 5D task target."""

    projector: OrientationProjector
    teleop: TeleopConfig = field(default_factory=TeleopConfig)

    _target: SO100TaskPose | None = field(default=None, init=False)
    _imu_reference: np.ndarray | None = field(default=None, init=False)
    _latched: SO100TaskPose | None = field(default=None, init=False)
    _was_engaged: bool = field(default=False, init=False)

    @property
    def target(self) -> SO100TaskPose | None:
        return self._target

    def reset(self) -> None:
        """Forget the latch. The next clutch press re-latches from measurement."""
        self._target = None
        self._imu_reference = None
        self._latched = None
        self._was_engaged = False

    def sync_target(self, pose: SO100TaskPose) -> None:
        """Adopt a downstream-clamped target without changing the clutch latch."""
        self._target = pose

    def force_target(self, pose: SO100TaskPose, sample: NintendoProSample | None = None) -> None:
        """Re-anchor teleop after a downstream safety hold.

        When a post-IK command is rejected, keeping the old internal target makes
        the next frame chase the same unsafe joint-space path again. Re-anchor
        both the frozen target and, if the clutch is held, the IMU reference to
        the controller's current attitude.
        """
        self._target = pose
        self._latched = pose
        if sample is not None and sample.clutch:
            self._imu_reference = quaternion_to_matrix(sample.imu_quaternion)
            self._was_engaged = True
        else:
            self._imu_reference = None
            self._was_engaged = False

    def _stick_velocity(self, sample: NintendoProSample) -> np.ndarray:
        """Stick deflections to metres of world translation for this step.

        The operator stands behind the arm, so the left stick is the horizontal
        plane in front of them and the right stick's vertical axis is height.
        Stick +Y reads as "up" on the device, which is forward on the table.
        """
        step = np.asarray(self.teleop.translation_step_m, float)
        gain = float(self.teleop.stick_translation_gain)
        return step * gain * np.array(
            [
                float(sample.left_stick[1]),  # forward/back  -> world +X
                float(-sample.left_stick[0]),  # left/right   -> world +Y
                float(sample.right_stick[1]),  # up/down      -> world +Z
            ]
        )

    def update(self, sample: NintendoProSample, measured: SO100TaskPose) -> RetargetResult:
        """Advance the target by one controller frame.

        Args:
            sample: The raw controller frame.
            measured: The arm's *measured* task pose, from forward kinematics of
                the joints actually read back. Used on the clutch's rising edge
                and at no other time.
        """
        latched_now = False

        if sample.clutch and not self._was_engaged:
            self._imu_reference = quaternion_to_matrix(sample.imu_quaternion)
            self._latched = measured
            self._target = measured
            latched_now = True
        elif self._target is None:
            self._target = measured

        self._was_engaged = sample.clutch

        if not sample.clutch:
            # Released: hold. Re-emitting the frozen target rather than stopping
            # the loop keeps IK correcting against servo droop.
            return RetargetResult(
                target=self._target,
                engaged=False,
                latched=False,
                delta_position_m=np.zeros(3),
                projected_pitch_delta=0.0,
                projected_roll_delta=0.0,
                rejected_rotation_norm=0.0,
            )

        delta_position = self._stick_velocity(sample)

        rotation_now = quaternion_to_matrix(sample.imu_quaternion)
        omega = rotation_log(rotation_now @ self._imu_reference.T) * self.teleop.rotation_gain

        # Projected at the *latched* pose, not the current target. Both give the
        # same axes to within the translation the operator has done since
        # clutching in, and holding them fixed for the duration of the hold is
        # what makes the gesture-to-attitude map single-valued: return the hand
        # to where it was and the gripper returns to where it was, whatever path
        # was taken in between. Re-evaluating each step would let a loop in the
        # gesture accumulate a permanent offset.
        projected = self.projector.project(omega, self._latched)

        self._target = SO100TaskPose(
            self._target.x + delta_position[0],
            self._target.y + delta_position[1],
            self._target.z + delta_position[2],
            self._latched.pitch + projected.delta_pitch,
            float(wrap_to_pi(self._latched.roll + projected.delta_roll)),
        )

        return RetargetResult(
            target=self._target,
            engaged=True,
            latched=latched_now,
            delta_position_m=delta_position,
            projected_pitch_delta=projected.delta_pitch,
            projected_roll_delta=projected.delta_roll,
            rejected_rotation_norm=projected.rejected_norm,
        )
