"""Kinematic simulation of the SO-100 arm, in MuJoCo.

## Which simulator, and why

MuJoCo 3.6, which is already in the development environment alongside lerobot.
Weighed against the alternatives for what this project actually needs:

| candidate | verdict |
|---|---|
| **MuJoCo** | already installed; loads the URDF with its meshes; exact analytic site Jacobians; mesh-level collision detection; offscreen cameras; a viewer |
| PyBullet | not installed, and nothing here it does better |
| Genesis / SAPIEN | not installed, heavy, aimed at photorealistic contact-rich RL we do not need yet |
| pinocchio + meshcat | both installed and excellent at kinematics, but no camera and no collision story, which is half of what we want a simulator for |

Three things this buys that `MockFollower` cannot:

1. **An independent check on our kinematics.** `ArmChain` is the control path's
   forward kinematics and Jacobian. MuJoCo computes both from the same URDF via
   entirely different code, and `scripts/check_kinematics_agreement.py` pins
   them -- together with placo -- to 1e-9 m. Three implementations agreeing is
   evidence; one implementation passing its own tests is not.
2. **Self-collision.** The feasibility atlas is built from the URDF, which has
   no collision reasoning at all, so it happily reports configurations where the
   gripper is inside the forearm. MuJoCo has the actual meshes.
3. **Cameras.** The blueprint records a wrist and a third-person view. Having
   them here means the dataset layout, the observation dict and the recording
   loop can be exercised and debugged before a USB camera is ever plugged in.

## Why kinematic and not dynamic

`qpos` is set directly and `mj_forward` is called; `mj_step` never is. The
question Phase 0 has to answer is geometric -- does the target chain produce
joint commands that put the gripper where it was asked, without collision --
and answering it kinematically is deterministic, exactly repeatable, and about
three orders of magnitude faster than integrating contact dynamics whose
parameters we would be inventing anyway. The STS3215's real behaviour is a
tracking lag, and that is modelled explicitly in `MujocoBackend` as a
first-order servo, the same model `MockFollower` uses, rather than emerging
from torque limits we have not measured.

When contact dynamics do start to matter -- grasping a block, not reaching for
one -- the model here is already the one to step.

## Rendering

Offscreen rendering needs a GL backend, chosen by the `MUJOCO_GL` environment
variable. On this machine `glfw` works and `egl`/`osmesa` are not installed, so
`render()` sets `MUJOCO_GL=glfw` if nothing has been chosen. On a headless
training box, install EGL and set `MUJOCO_GL=egl` before importing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import GRIPPER_JOINT, REPO_ROOT, ArmConfig, TeleopConfig

DEFAULT_MODEL_PATH = REPO_ROOT / "assets" / "mujoco" / "so100.xml"

TCP_SITE = "tcp"
TOOL_SITE = "tool"


@dataclass(frozen=True)
class SelfCollision:
    """One pair of arm links found intersecting."""

    body_a: str
    body_b: str
    depth_m: float  # penetration depth; MuJoCo reports these as negative distances

    def __str__(self) -> str:
        return f"{self.body_a} <-> {self.body_b} ({1000 * self.depth_m:.2f} mm)"


class MujocoArm:
    """The arm as a kinematic model: set joint angles, read frames and contacts."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        arm: ArmConfig | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.arm = arm or ArmConfig()
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.model_path}. "
                "Generate it with scripts/build_mujoco_model.py"
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._renderer = None

        self.joint_names = (*self.arm.joint_names, GRIPPER_JOINT)
        self._qpos_index = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in self.joint_names
            ]
        )
        if np.any(self._qpos_index < 0):
            raise ValueError(f"model {self.model_path} is missing one of {self.joint_names}")

        self._site_id = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in (TCP_SITE, TOOL_SITE)
        }
        self.reset_home()

    # ----------------------------------------------------------------- state

    def reset_home(self) -> None:
        """Return to the model's `home` keyframe, which is `TeleopConfig.home_joints_deg`."""
        self._mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self._forward()

    def _forward(self) -> None:
        # mj_forward, not mj_step: positions and contacts, no integration.
        self._mujoco.mj_forward(self.model, self.data)

    def set_joints_deg(self, joints_deg: np.ndarray) -> None:
        """Place the arm at a configuration. Gripper optional."""
        values = np.deg2rad(np.asarray(joints_deg, float))
        self.data.qpos[self._qpos_index[: len(values)]] = values
        self._forward()

    def joints_deg(self) -> np.ndarray:
        """Current configuration, arm joints then gripper."""
        return np.rad2deg(self.data.qpos[self._qpos_index])

    # ----------------------------------------------------------------- frames

    def site_pose(self, site: str) -> np.ndarray:
        """World pose of a named site, 4x4."""
        index = self._site_id[site]
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[index].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[index]
        return T

    def tcp_pose(self) -> np.ndarray:
        """World pose of the URDF TCP frame — comparable with `ArmChain.fk`."""
        return self.site_pose(TCP_SITE)

    def tool_pose(self) -> np.ndarray:
        """World pose of the task-space tool frame — comparable with `TaskIK5D.forward`."""
        return self.site_pose(TOOL_SITE)

    def jacobian(self, site: str = TOOL_SITE) -> np.ndarray:
        """MuJoCo's own geometric Jacobian at a site, 6x5, world frame.

        Independent of `ArmChain.jacobian` in every line of code that produces
        it, which is the point of having it.
        """
        n = self.model.nv
        jacp, jacr = np.zeros((3, n)), np.zeros((3, n))
        self._mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site_id[site])
        columns = self.model.jnt_dofadr[
            [
                self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.arm.joint_names
            ]
        ]
        return np.vstack([jacp[:, columns], jacr[:, columns]])

    # ------------------------------------------------------------- collision

    def self_collisions(self) -> list[SelfCollision]:
        """Link pairs currently intersecting, excluding adjacent ones.

        The model excludes contacts between neighbouring links, which always
        touch at their joint, and gives the table no collision geometry, since a
        base resting on it would be a permanent false positive. What is left is
        the arm folding into itself -- which the URDF-derived feasibility atlas
        has no way of knowing about, and which is the single most useful thing
        this simulation says that nothing else in the project can.
        """
        found = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            bodies = [
                self._mujoco.mj_id2name(
                    self.model,
                    self._mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[geom]),
                )
                for geom in (contact.geom1, contact.geom2)
            ]
            found.append(SelfCollision(bodies[0], bodies[1], float(contact.dist)))
        return found

    def in_collision(self) -> bool:
        return self.data.ncon > 0

    # --------------------------------------------------------------- rendering

    def render(self, camera: str = "third_person", width: int = 640, height: int = 480):
        """Render one frame from a named camera as an `(H, W, 3)` uint8 array."""
        if self._renderer is None or (
            self._renderer.height != height or self._renderer.width != width
        ):
            os.environ.setdefault("MUJOCO_GL", "glfw")
            self._renderer = self._mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class MujocoBackend:
    """`RobotBackend` over the MuJoCo model, so the teleop loop can drive it.

    Interchangeable with `MockFollower` and `SOFollowerBackend`: same interface,
    same servo lag model, so what the loop does here is what it will do on the
    arm. What it adds over the mock is that the joints it is commanded to are
    checked against real geometry -- `collisions` accumulates every step that
    put the arm inside itself, which a first-order lag on five scalars cannot
    notice.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        arm: ArmConfig | None = None,
        *,
        tracking_gain: float = 0.35,
        read_noise_deg: float = 0.05,
        initial_joints_deg: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.arm = arm or ArmConfig()
        self.sim = MujocoArm(model_path, self.arm)
        self.tracking_gain = tracking_gain
        self.read_noise_deg = read_noise_deg
        self.collisions: list[tuple[int, SelfCollision]] = []
        self.write_count = 0

        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._lo = np.array([self.arm.joint_limits_deg[j][0] for j in self.joint_names])
        self._hi = np.array([self.arm.joint_limits_deg[j][1] for j in self.joint_names])

        if initial_joints_deg is None:
            teleop = TeleopConfig()
            gripper_mid = sum(self.arm.joint_limits_deg[GRIPPER_JOINT]) / 2.0
            initial_joints_deg = np.array([*teleop.home_joints_deg, gripper_mid])
        self._position = np.asarray(initial_joints_deg, float).copy()
        self.sim.set_joints_deg(self._position)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return (*self.arm.joint_names, GRIPPER_JOINT)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _require_connection(self) -> None:
        if not self._connected:
            raise RuntimeError("MujocoBackend is not connected; call connect() first")

    def read_joints_deg(self) -> np.ndarray:
        self._require_connection()
        noise = self._rng.normal(0.0, self.read_noise_deg, size=self._position.shape)
        return np.clip(self._position + noise, self._lo, self._hi)

    def write_joints_deg(self, target_deg: np.ndarray) -> None:
        self._require_connection()
        target = np.clip(np.asarray(target_deg, float), self._lo, self._hi)
        if target.shape != self._position.shape:
            raise ValueError(
                f"expected {self._position.shape[0]} joint values in order {self.joint_names}, "
                f"got shape {target.shape}"
            )
        self._position = self._position + self.tracking_gain * (target - self._position)
        self.sim.set_joints_deg(self._position)
        for collision in self.sim.self_collisions():
            self.collisions.append((self.write_count, collision))
        self.write_count += 1

    def true_joints_deg(self) -> np.ndarray:
        """Noise-free state. Available in simulation only, for test assertions."""
        return self._position.copy()
