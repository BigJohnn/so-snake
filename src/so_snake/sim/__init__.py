"""Kinematic simulation of the SO-100, in MuJoCo.

Why MuJoCo, and why only kinematics, is in `mujoco_arm.MujocoArm`.
"""

from .mujoco_arm import DEFAULT_MODEL_PATH, MujocoArm, MujocoBackend, SelfCollision

__all__ = ["DEFAULT_MODEL_PATH", "MujocoArm", "MujocoBackend", "SelfCollision"]
