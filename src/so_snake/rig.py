"""Building a backend and a teleoperation source from a description of one.

Three things now need to say "give me the MuJoCo arm driven by a scripted
waveform" or "give me the real arm driven by the Pro controller" -- the record
script, the replay script, and the GUI gateway. Without a shared factory each
would grow its own copy of the joint-map loading, the safety clamps and the
import fallbacks, and they would drift. The point of the backend Protocol is
that the choice is one line; this keeps it that way.

What this module deliberately does *not* do is safety choreography. Moving to
the start pose, settling under torque, confirming with the operator: those are
`scripts/teleop_real_arm.py`'s, they are specific to what the caller is about to
do, and burying them in a constructor would make them easy to skip.

`availability()` answers what this machine can actually run, so a UI can grey
out what is missing instead of offering it and failing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ARM_JOINTS, GRIPPER_JOINT, REPO_ROOT, SoSnakeConfig
from .m4_execution.backends import MockFollower, RobotBackend
from .teleop.sources import ScriptedSource, TeleopSource

DEFAULT_JOINT_MAP = REPO_ROOT / "assets" / "so100_joint_map.json"

BACKENDS = ("mock", "mujoco", "real")
SOURCES = ("scripted", "pro")

# Backends that command a physical arm. Everything that has to be more careful
# -- confirmations, conservative defaults, warnings in the UI -- keys off this
# rather than off the backend name, so adding a second real backend later does
# not mean auditing every caller.
PHYSICAL_BACKENDS = frozenset({"real"})


@dataclass(frozen=True)
class RigSpec:
    """Which arm, driven by which input, with which safety settings."""

    backend: str = "mock"
    source: str = "scripted"

    # -- real arm only ------------------------------------------------------
    port: str = ""
    robot_id: str = "so_snake"
    joint_map_path: Path = DEFAULT_JOINT_MAP
    # lerobot's hardware clamp, degrees per servo write. Behind the loop's own
    # `max_joint_step_deg`; the gripper gets a multiple of it because it is
    # commanded absolutely and this is its only speed knob.
    max_relative_target_deg: float = 5.0
    gripper_speed_mult: float = 3.0

    # -- controller only ----------------------------------------------------
    device_id: int | None = None

    # -- scripted source only -----------------------------------------------
    scripted_steps: int = 1800
    scripted_amplitude: float = 0.2
    scripted_rotation_amplitude_rad: float = 0.10
    scripted_loop: bool = True

    @property
    def is_physical(self) -> bool:
        return self.backend in PHYSICAL_BACKENDS

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.backend == "real" and not self.port:
            raise ValueError("the real backend needs --port")
        if self.max_relative_target_deg <= 0 or self.gripper_speed_mult <= 0:
            raise ValueError("max_relative_target_deg and gripper_speed_mult must be positive")


def build_backend(spec: RigSpec, config: SoSnakeConfig | None = None) -> RobotBackend:
    """Construct the backend `spec` names. Does not connect it."""
    spec.validate()
    config = config or SoSnakeConfig()

    if spec.backend == "mock":
        return MockFollower(arm=config.arm)

    if spec.backend == "mujoco":
        error = mujoco_import_error()
        if error:
            # Raised as a plain message rather than letting the original
            # AttributeError out: the caller sees "no attribute EGLDeviceEXT",
            # which says nothing about what to do. The GL backend is the usual
            # culprit and is fixable from the shell.
            raise RuntimeError(
                f"MuJoCo is installed but will not import here -- {error}. "
                "This is usually the GL backend: try MUJOCO_GL=glfw (needs a display) "
                "or upgrade PyOpenGL for EGL (headless)."
            )
        from .sim import MujocoBackend

        return MujocoBackend(arm=config.arm)

    from .m4_execution import JointFrameMap, SOFollowerBackend

    map_path = Path(spec.joint_map_path)
    if not map_path.is_file():
        # Refusing is the safe failure. `SOFollowerBackend(joint_map=None)`
        # passes lerobot's calibration frame straight through as if it were the
        # URDF frame, which drives the arm into itself.
        raise FileNotFoundError(
            f"joint map not found: {map_path}. Build it with "
            "scripts/map_joint_frames.py draft / signs / check before driving the arm."
        )
    clamp: dict[str, float] = {j: spec.max_relative_target_deg for j in ARM_JOINTS}
    clamp[GRIPPER_JOINT] = spec.max_relative_target_deg * spec.gripper_speed_mult
    return SOFollowerBackend(
        port=spec.port,
        arm=config.arm,
        robot_id=spec.robot_id,
        max_relative_target=clamp,
        joint_map=JointFrameMap.load(map_path),
    )


def build_source(spec: RigSpec, config: SoSnakeConfig | None = None) -> TeleopSource:
    """Construct the teleoperation source `spec` names. Does not connect it."""
    spec.validate()

    if spec.source == "scripted":
        source = ScriptedSource.from_waveform(
            n_steps=spec.scripted_steps,
            amplitude=spec.scripted_amplitude,
            rotation_amplitude_rad=spec.scripted_rotation_amplitude_rad,
        )
        # A GUI session runs until the operator stops it, so the waveform loops
        # rather than emitting "stop" when it runs out. The CLI gate wants the
        # opposite and passes `scripted_loop=False`.
        source.loop = spec.scripted_loop
        return source

    from .teleop.sources import NintendoProSource

    return NintendoProSource(controller="pro", device_id=spec.device_id)


def _importable(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def mujoco_import_error() -> str:
    """Actually import mujoco, and report why it failed if it did.

    `find_spec` is not enough here. MuJoCo reads `MUJOCO_GL` at import time and
    eagerly imports the matching GL context module, so an installed mujoco can
    still fail to import -- an old PyOpenGL with `MUJOCO_GL=egl` raises
    `AttributeError: module 'OpenGL.EGL' has no attribute 'EGLDeviceEXT'`.
    Reporting the package as present on the strength of the spec alone means
    offering the operator a backend that 500s when they select it.

    Cached by `sys.modules` after the first success, so this costs one import.
    """
    try:
        import mujoco  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - the reason is the payload
        return f"{type(exc).__name__}: {exc}"
    return ""


def availability() -> dict[str, Any]:
    """What this machine can run right now, and why not where it cannot.

    Checked by import rather than by trying to construct anything: probing a
    serial port opens it, and a GUI polling its own capability list must not be
    touching the arm's bus.
    """
    # mujoco is imported for real (see `mujoco_import_error`); lerobot is not,
    # because importing it pulls in torch and this is polled by a config page.
    mujoco_error = mujoco_import_error() if _importable("mujoco") else "mujoco not installed (.[sim] extra)"
    has_lerobot = _importable("lerobot")
    has_servo_sdk = _importable("scservo_sdk")
    joint_map_exists = DEFAULT_JOINT_MAP.is_file()

    def entry(ok: bool, reason: str = "") -> dict[str, Any]:
        return {"available": ok, "reason": "" if ok else reason}

    return {
        "backends": {
            "mock": entry(True),
            "mujoco": entry(not mujoco_error, mujoco_error),
            "real": entry(
                has_lerobot and has_servo_sdk and joint_map_exists,
                "; ".join(
                    filter(
                        None,
                        [
                            "" if has_lerobot else "lerobot not installed (.[teleop] extra)",
                            "" if has_servo_sdk else "feetech-servo-sdk not installed",
                            "" if joint_map_exists else f"joint map missing at {DEFAULT_JOINT_MAP}",
                        ],
                    )
                ),
            ),
        },
        "sources": {
            "scripted": entry(True),
            "pro": entry(has_lerobot, "lerobot not installed (.[teleop] extra)"),
        },
        "joint_map_path": str(DEFAULT_JOINT_MAP),
        "joint_map_present": joint_map_exists,
    }
