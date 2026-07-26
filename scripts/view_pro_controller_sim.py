#!/usr/bin/env python
"""Drive the MuJoCo SO-100 model from a USB Nintendo Pro Controller with a viewer.

Run from the repo root:
    # Linux
    PYTHONPATH=src /home/hanyu/Codes/lerobot/.venv/bin/python scripts/view_pro_controller_sim.py
    # macOS (Apple Silicon / Intel): the passive viewer must own the main
    # thread, so it runs under `mjpython`. Launching with plain `python` still
    # works -- this script re-executes itself under `mjpython` automatically.
    PYTHONPATH=src .venv/bin/python scripts/view_pro_controller_sim.py

Controls come from lerobot's NintendoTeleop mapping:
  - left stick X/Y: task-space X/Y
  - right stick Y: task-space Z
  - ZL: IMU rotation clutch
  - R: close gripper incrementally

Close the MuJoCo viewer or press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from so_snake.config import SoSnakeConfig
from so_snake.sim import MujocoBackend
from so_snake.teleop import (
    NintendoProSample,
    NintendoProSource,
    ScriptedSource,
    TeleopLoop,
    TeleopSource,
)


class StoppableSource:
    """Wrap a teleop source so the viewer thread can stop `TeleopLoop.run`."""

    def __init__(self, source: TeleopSource, stop_event: threading.Event) -> None:
        self._source = source
        self._stop_event = stop_event

    def connect(self) -> None:
        self._source.connect()

    def disconnect(self) -> None:
        self._source.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._source.is_connected

    def read(self) -> NintendoProSample:
        if self._stop_event.is_set():
            return NintendoProSample(
                t=0.0,
                left_stick=np.zeros(2),
                right_stick=np.zeros(2),
                imu_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                clutch=False,
                events=frozenset({"stop"}),
            )
        return self._source.read()


class ViewerLockedBackend:
    """Guard MuJoCo data access while the passive viewer is rendering it."""

    def __init__(self, backend: MujocoBackend) -> None:
        self._backend = backend
        self._lock_factory: Callable[[], object] | None = None

    def set_viewer_lock(self, lock_factory: Callable[[], object]) -> None:
        self._lock_factory = lock_factory

    def _guard(self):
        if self._lock_factory is None:
            return nullcontext()
        return self._lock_factory()

    @property
    def sim(self):
        return self._backend.sim

    @property
    def collisions(self):
        return self._backend.collisions

    @property
    def write_count(self) -> int:
        return self._backend.write_count

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._backend.joint_names

    def connect(self) -> None:
        with self._guard():
            self._backend.connect()

    def disconnect(self) -> None:
        with self._guard():
            self._backend.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._backend.is_connected

    def read_joints_deg(self) -> np.ndarray:
        with self._guard():
            return self._backend.read_joints_deg()

    def write_joints_deg(self, target_deg: np.ndarray) -> None:
        with self._guard():
            self._backend.write_joints_deg(target_deg)

    def robot_mesh_min_z(self) -> tuple[float, str]:
        with self._guard():
            return self._backend.robot_mesh_min_z()

    def command_robot_mesh_min_z_deg(self, target_deg: np.ndarray) -> tuple[float, str]:
        with self._guard():
            return self._backend.command_robot_mesh_min_z_deg(target_deg)


@dataclass
class WorkerState:
    error: BaseException | None = None
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_error(self, exc: BaseException) -> None:
        with self.lock:
            self.error = exc

    def set_done(self) -> None:
        with self.lock:
            self.done = True

    def snapshot(self) -> tuple[BaseException | None, bool]:
        with self.lock:
            return self.error, self.done



class TeleopMonitor:
    """Emit sparse warning records for unsafe-looking teleop transients."""

    def __init__(
        self,
        *,
        warn_z_m: float,
        warn_raw_imu_step_rad: float,
        warn_task_rot_step_rad: float,
        log_path: str | None,
    ) -> None:
        self.warn_z_m = warn_z_m
        self.warn_raw_imu_step_rad = warn_raw_imu_step_rad
        self.warn_task_rot_step_rad = warn_task_rot_step_rad
        self.log_path = log_path
        self._last_index = -1
        self._last_raw_quat: np.ndarray | None = None
        self._last_command: np.ndarray | None = None
        self._last_collision_count = 0
        self._last_reason_key: tuple | None = None
        self._same_reason_count = 0

    @staticmethod
    def _as_list(value) -> list[float]:
        return [float(v) for v in np.asarray(value, dtype=float).reshape(-1)]

    @staticmethod
    def _quat_step_rad(a: np.ndarray | None, b: np.ndarray) -> float:
        if a is None:
            return 0.0
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        dot = float(np.clip(abs(np.dot(a / na, b / nb)), -1.0, 1.0))
        return float(2.0 * np.arccos(dot))

    def check(self, loop: TeleopLoop, backend: ViewerLockedBackend) -> None:
        records = loop.stats.records
        if not records:
            return
        cap = float(loop.config.teleop.max_joint_step_deg)
        lo, hi = loop.config.arm.limits_deg_array()
        near_limit_margin_deg = 1.0
        new_collision_count = len(backend.collisions)
        for record in records[self._last_index + 1 :]:
            raw_quat = np.asarray(record.raw["action.raw.imu_quaternion"], dtype=float)
            raw_imu_step = self._quat_step_rad(self._last_raw_quat, raw_quat)
            self._last_raw_quat = raw_quat

            command = np.asarray(record.commanded_joints_deg, dtype=float)
            command_step = 0.0
            if self._last_command is not None:
                command_step = float(np.abs(command - self._last_command).max())
            self._last_command = command

            task_rot_step = float(np.linalg.norm(record.task_delta[3:5]))
            target_z = float(record.task_target[2])
            achieved_z = float(record.achieved_position[2])
            robot_mesh_min_z = getattr(record, "robot_mesh_min_z_m", None)
            robot_mesh_min_body = str(getattr(record, "robot_mesh_min_body", ""))
            reasons: list[str] = []
            if target_z <= self.warn_z_m:
                reasons.append("low_target_z")
            if achieved_z <= self.warn_z_m:
                reasons.append("low_achieved_z")
            if robot_mesh_min_z is not None and robot_mesh_min_z <= loop.config.teleop.min_robot_mesh_z_m:
                reasons.append("low_robot_mesh_z")
            if raw_imu_step >= self.warn_raw_imu_step_rad:
                reasons.append("raw_imu_jump")
            if task_rot_step >= self.warn_task_rot_step_rad:
                reasons.append("task_rot_step_high")
            if command_step >= 0.8 * cap:
                reasons.append("joint_step_near_cap")
            near_limit = (command <= lo + near_limit_margin_deg) | (command >= hi - near_limit_margin_deg)
            if bool(near_limit.any()):
                reasons.append("joint_near_limit")
            if not record.ik_solver_converged:
                reasons.append("ik_solver_not_converged")
            if record.joint_rate_clamped:
                reasons.append("joint_rate_clamped")
            if getattr(record, "command_safety_held", False):
                reasons.append("command_safety_held")
            if record.joint_limit_clamped:
                reasons.append("joint_limit_clamped")
            if record.workspace_clamped:
                reasons.append("workspace_clamped")
            if record.atlas_pitch_clamped:
                reasons.append("atlas_pitch_clamped")
            if record.atlas_roll_infeasible:
                reasons.append("atlas_roll_infeasible")
            if new_collision_count > self._last_collision_count:
                reasons.append("new_self_collision")

            if reasons:
                payload = {
                    "step": int(record.index),
                    "t": float(record.t),
                    "reasons": reasons,
                    "clutch": bool(record.clutch_engaged),
                    "target": self._as_list(record.task_target),
                    "task_delta": self._as_list(record.task_delta),
                    "achieved_position": self._as_list(record.achieved_position),
                    "robot_mesh_min_z_m": None if robot_mesh_min_z is None else float(robot_mesh_min_z),
                    "robot_mesh_min_body": robot_mesh_min_body,
                    "robot_mesh_min_z_limit_m": float(loop.config.teleop.min_robot_mesh_z_m),
                    "raw_sticks": self._as_list(record.raw["action.raw.sticks"]),
                    "raw_imu_quaternion": self._as_list(raw_quat),
                    "raw_imu_step_rad": raw_imu_step,
                    "task_rot_step_rad": task_rot_step,
                    "command_step_deg": command_step,
                    "command_safety_held": bool(getattr(record, "command_safety_held", False)),
                    "command_safety_reason": str(getattr(record, "command_safety_reason", "")),
                    "commanded_joints_deg": self._as_list(command),
                    "joint_near_limit": [bool(v) for v in near_limit],
                    "joint_limit_margin_deg": self._as_list(np.minimum(command - lo, hi - command)),
                    "measured_joints_deg": self._as_list(record.measured_joints_deg),
                    "ik_position_error_m": float(record.ik_position_error_m),
                    "ik_pitch_error_rad": float(record.ik_pitch_error_rad),
                    "ik_roll_error_rad": float(record.ik_roll_error_rad),
                    "projected_pitch_delta": float(record.projected_pitch_delta),
                    "projected_roll_delta": float(record.projected_roll_delta),
                    "rejected_rotation_norm": float(record.rejected_rotation_norm),
                    "yaw_residual_rad": float(record.yaw_residual_rad),
                    "flags": {
                        "workspace_clamped": bool(record.workspace_clamped),
                        "atlas_pitch_clamped": bool(record.atlas_pitch_clamped),
                        "atlas_roll_infeasible": bool(record.atlas_roll_infeasible),
                        "joint_limit_clamped": bool(record.joint_limit_clamped),
                        "joint_rate_clamped": bool(record.joint_rate_clamped),
                        "command_safety_held": bool(getattr(record, "command_safety_held", False)),
                        "ik_solver_converged": bool(record.ik_solver_converged),
                        "ik_reseeded": bool(record.ik_reseeded),
                    },
                    "collision_count": int(new_collision_count),
                }
                reason_key = (tuple(reasons), tuple(round(float(v), 3) for v in record.task_target))
                if reason_key == self._last_reason_key:
                    self._same_reason_count += 1
                else:
                    self._last_reason_key = reason_key
                    self._same_reason_count = 1
                urgent = any(
                    reason in reasons
                    for reason in (
                        "low_target_z",
                        "low_achieved_z",
                        "raw_imu_jump",
                        "task_rot_step_high",
                        "joint_step_near_cap",
                        "new_self_collision",
                        "command_safety_held",
                        "low_robot_mesh_z",
                    )
                )
                should_print = urgent or self._same_reason_count in (1, 30) or self._same_reason_count % 300 == 0
                if should_print:
                    repeat = "" if self._same_reason_count == 1 else f" repeat={self._same_reason_count}"
                    print(
                        "WARN teleop "
                        f"step={record.index} reasons={','.join(reasons)}{repeat} "
                        f"target_z={target_z:.3f} achieved_z={achieved_z:.3f} "
                        f"mesh_z={robot_mesh_min_z if robot_mesh_min_z is not None else float('nan'):.3f} "
                        f"mesh_body={robot_mesh_min_body or '-'} "
                        f"raw_imu_step={raw_imu_step:.3f} task_rot_step={task_rot_step:.3f} "
                        f"joint_step={command_step:.2f}",
                        flush=True,
                    )
                if self.log_path and should_print:
                    payload["repeat"] = int(self._same_reason_count)
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, sort_keys=True) + "\n")
        self._last_index = records[-1].index
        self._last_collision_count = new_collision_count


def _print_status(loop: TeleopLoop, backend: ViewerLockedBackend) -> None:
    if not loop.stats.records:
        print("status: waiting for controller samples", flush=True)
        return
    record = loop.stats.records[-1]
    sticks = record.raw["action.raw.sticks"]
    try:
        mesh_z, mesh_body = backend.robot_mesh_min_z()
        mesh_text = f" mesh_z={mesh_z:.3f} mesh_body={mesh_body or '-'}"
    except Exception:
        mesh_text = ""
    print(
        "status: "
        f"step={record.index + 1} "
        f"clutch={int(record.clutch_engaged)} "
        f"sticks=[lx={sticks[0]:+.2f} ly={sticks[1]:+.2f} rz={sticks[3]:+.2f}] "
        f"writes={backend.write_count} "
        f"collisions={len(backend.collisions)}"
        f"{mesh_text}",
        flush=True,
    )


def _print_final(loop: TeleopLoop, backend: ViewerLockedBackend) -> None:
    print()
    print("=" * 66)
    print("Viewer Teleop Summary")
    print("=" * 66)
    records = loop.stats.records
    print(f"  steps             {len(records)}")
    print(f"  writes issued     {backend.write_count}")
    print(f"  self-collisions   {len(backend.collisions)}")
    if records:
        summary = loop.stats.summary()
        clutch = np.array([record.clutch_engaged for record in records], dtype=bool)
        sticks = np.array([record.raw["action.raw.sticks"] for record in records])
        print(f"  clutch engaged    {100.0 * clutch.mean():6.1f} %")
        print(f"  stick abs max     {np.abs(sticks).max(axis=0)}")
        print(f"  loop hz median    {summary['loop_hz_median']:8.1f}")
        print(f"  IK pos err p95    {summary['ik_pos_err_p95_mm']:8.4f} mm")


def _run_loop(loop: TeleopLoop, max_steps: int | None, state: WorkerState) -> None:
    try:
        loop.run(max_steps=max_steps, realtime=True)
    except BaseException as exc:
        state.set_error(exc)
    finally:
        state.set_done()


def _reexec_under_mjpython_if_needed() -> None:
    """On macOS relaunch the process under `mjpython`, which the viewer requires.

    MuJoCo's `launch_passive` refuses to run on macOS unless the Cocoa UI owns
    the main thread, which the `mjpython` launcher -- shipped in the same `bin/`
    as the active interpreter -- arranges. We detect that we are *not* already
    under it via `mujoco.viewer._MJPYTHON`, the same flag MuJoCo checks before
    raising, then re-exec this script with its arguments through mjpython. The
    environment (so `PYTHONPATH=src`) is inherited across the exec.
    """
    if sys.platform != "darwin":
        return
    try:
        import mujoco.viewer
    except ImportError:
        return  # let the main import path report the missing dependency
    if getattr(mujoco.viewer, "_MJPYTHON", None) is not None:
        return  # already under mjpython

    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        found = shutil.which("mjpython")
        mjpython = Path(found) if found else None
    if mjpython is None:
        print(
            "error: on macOS the MuJoCo viewer must run under `mjpython`, which was not "
            "found next to this interpreter or on PATH.\n"
            'Install the sim extra (uv pip install -e ".[sim]") and rerun.',
            file=sys.stderr,
        )
        raise SystemExit(2)

    script = os.path.abspath(__file__)
    os.execv(str(mjpython), [str(mjpython), script, *sys.argv[1:]])


def _scripted_source() -> ScriptedSource:
    """A hardware-free waveform that sweeps the workspace, looping forever.

    This is the fallback when no Switch Pro controller / lerobot is present --
    the default situation on macOS -- so the viewer still shows the arm moving.
    """
    source = ScriptedSource.from_waveform(600, rotation_amplitude_rad=0.10)
    source.loop = True
    return source


def _build_teleop_source(kind: str, device_id: int | None) -> tuple[TeleopSource, str]:
    """Pick the teleop input for this machine.

    - ``pro``      force the real Switch Pro controller (lerobot + hidapi).
    - ``scripted`` force the deterministic waveform; needs no hardware, runs anywhere.
    - ``auto``     try the controller, fall back to the waveform when lerobot or the
      device is unavailable -- the macOS / no-controller case.

    The controller is *probed by connecting*: lerobot is imported and the HID
    device opened up front, so ``auto`` can catch a missing dependency or an
    unplugged pad here instead of failing deep inside the viewer's worker thread.
    """

    def build_pro() -> TeleopSource:
        source = NintendoProSource(controller="pro", device_id=device_id)
        source.connect()  # raises here if lerobot / the device is unavailable
        return source

    if kind == "scripted":
        return _scripted_source(), "scripted waveform (no hardware)"
    if kind == "pro":
        return build_pro(), "Switch Pro controller"

    # auto
    try:
        return build_pro(), "Switch Pro controller"
    except Exception as exc:
        print(
            f"note: no Switch Pro controller available ({type(exc).__name__}: {exc}); "
            "using the scripted waveform.",
            file=sys.stderr,
        )
        print(
            "      to teleoperate: install lerobot main "
            "(pip install 'lerobot @ git+https://github.com/huggingface/lerobot.git') "
            "plus a Pro controller, then pass --source pro.",
            file=sys.stderr,
        )
        return _scripted_source(), "scripted waveform (no hardware)"


def main() -> int:
    _reexec_under_mjpython_if_needed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("auto", "pro", "scripted"),
        default="auto",
        help="teleop input: auto-detect (default), force the Pro controller, or the scripted waveform",
    )
    parser.add_argument("--device-id", type=int, default=None, help="optional lerobot NintendoTeleop device id")
    parser.add_argument("--steps", type=int, default=None, help="stop after this many control frames")
    parser.add_argument("--print-every", type=float, default=1.0, help="seconds between terminal status lines")
    parser.add_argument("--mujoco-gl", default=None, help="override MUJOCO_GL, e.g. glfw, egl, osmesa")
    parser.add_argument("--warn-z", type=float, default=0.005, help="warn when target/achieved TCP z is below this")
    parser.add_argument("--warn-raw-imu-step", type=float, default=0.35, help="warn on raw IMU quaternion step above this many radians")
    parser.add_argument("--warn-task-rot-step", type=float, default=0.12, help="warn on task pitch/roll step above this many radians")
    parser.add_argument("--warn-log", default="/tmp/so_snake_teleop_warnings.jsonl", help="JSONL warning log path; empty disables file logging")
    args = parser.parse_args()

    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.print_every < 0.0:
        raise SystemExit("--print-every must be non-negative")
    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    warn_log = args.warn_log or None
    if warn_log:
        Path(warn_log).parent.mkdir(parents=True, exist_ok=True)

    try:
        import mujoco.viewer

        config = SoSnakeConfig()
        stop_event = threading.Event()
        inner_source, source_label = _build_teleop_source(args.source, args.device_id)
        source = StoppableSource(inner_source, stop_event)
        backend = ViewerLockedBackend(MujocoBackend(arm=config.arm))
        loop = TeleopLoop(source, backend, config)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\nThis script needs MuJoCo and a graphical display; the Pro controller "
            "additionally needs lerobot + hidapi. Try --source scripted to run without a controller.",
            file=sys.stderr,
        )
        return 2

    state = WorkerState()
    monitor = TeleopMonitor(
        warn_z_m=args.warn_z,
        warn_raw_imu_step_rad=args.warn_raw_imu_step,
        warn_task_rot_step_rad=args.warn_task_rot_step,
        log_path=warn_log,
    )
    worker = threading.Thread(target=_run_loop, args=(loop, args.steps, state), daemon=True)

    print(f"Opening MuJoCo viewer. Teleop source: {source_label}.")
    if args.source != "scripted" and source_label.startswith("Switch"):
        print("Use left stick / right stick Y; hold ZL for IMU rotation.")
    else:
        print("Scripted sweep is driving the arm; no controller input needed.")
    print("Close the viewer or press Ctrl-C to stop.")
    if warn_log:
        print(f"Teleop warning log: {warn_log}")

    try:
        with mujoco.viewer.launch_passive(backend.sim.model, backend.sim.data) as viewer:
            backend.set_viewer_lock(viewer.lock)
            with viewer.lock():
                viewer.cam.distance = 0.75
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -25.0
                viewer.cam.lookat[:] = np.array([0.12, 0.0, 0.12])

            worker.start()
            next_print = time.monotonic() + args.print_every
            while viewer.is_running():
                error, done = state.snapshot()
                if error is not None:
                    raise error
                if done:
                    break

                viewer.sync()
                monitor.check(loop, backend)
                time.sleep(1.0 / 60.0)

                if args.print_every > 0.0 and time.monotonic() >= next_print:
                    _print_status(loop, backend)
                    next_print = time.monotonic() + args.print_every
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        try:
            source.disconnect()
        except Exception:
            pass
        try:
            backend.disconnect()
        except Exception:
            pass
        backend.sim.close()

    error, _done = state.snapshot()
    if error is not None and not isinstance(error, KeyboardInterrupt):
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2

    _print_final(loop, backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
