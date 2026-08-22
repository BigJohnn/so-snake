#!/usr/bin/env python
"""Run a locally trained ACT or pi0.5 policy through so-snake's task-space safety layer."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from so_snake.config import SoSnakeConfig
from so_snake.data.export import apply_action, crop_image, validate_roi
from so_snake.devices import detect_arm_port
from so_snake.m0_perception import CameraSpec
from so_snake.m3_safety.atlas import DEFAULT_ATLAS_PATH, FeasibilityAtlas
from so_snake.m3_safety.ik5d import TaskIK5D
from so_snake.m3_safety.task_pose import SO100TaskPose, TaskPoseTracker
from so_snake.pacing import RateKeeper
from so_snake.rig import RigSpec, build_backend, build_cameras


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", required=True, help="language task supplied to pi0.5")
    p.add_argument("--backend", choices=("mock", "mujoco", "real"), default="mock")
    p.add_argument("--port", default="")
    p.add_argument("--camera", action="append", default=[], metavar="ROLE=DEVICE")
    p.add_argument("--roi", action="append", default=[], metavar="ROLE=X,Y,W,H",
                   help="normalised crop carried from the exported training dataset")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--device", default="")
    p.add_argument("--action-space", choices=("delta", "absolute"), default="delta")
    p.add_argument("--max-relative-target", type=float, default=5.0)
    return p.parse_args()


def main() -> int:
    a = args(); cfg = SoSnakeConfig()
    camera_specs = []
    for item in a.camera:
        role, sep, device = item.partition("=")
        if not sep or not role or not device:
            raise ValueError("--camera must be ROLE=DEVICE")
        camera_specs.append(CameraSpec(role, int(device) if device.lstrip("-").isdigit() else device))
    rois: dict[str, tuple[float, float, float, float]] = {}
    for item in a.roi:
        role, sep, values = item.partition("=")
        if not sep or not role:
            raise ValueError("--roi must be ROLE=X,Y,W,H")
        rois[role] = validate_roi(values.split(","), label=f"ROI for {role}")
    port = detect_arm_port(a.port) if a.backend == "real" else a.port
    spec = RigSpec(backend=a.backend, port=port, cameras=tuple(camera_specs), max_relative_target_deg=a.max_relative_target)
    backend, cameras = build_backend(spec, cfg), build_cameras(spec)
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.control_utils import predict_action
        from lerobot.utils.utils import get_safe_torch_device
        policy_cfg = PreTrainedConfig.from_pretrained(a.checkpoint)
        if a.device: policy_cfg.device = a.device
        policy = get_policy_class(policy_cfg.type).from_pretrained(a.checkpoint, config=policy_cfg)
        pre, post = make_pre_post_processors(policy_cfg, pretrained_path=str(a.checkpoint))
        device = get_safe_torch_device(policy_cfg.device)
        ik = TaskIK5D(arm=cfg.arm, teleop=cfg.teleop, ik=cfg.ik)
        tracker = TaskPoseTracker(cfg.limits, home=ik.task_pose(np.asarray(cfg.teleop.home_joints_deg)).pose)
        atlas = FeasibilityAtlas.load(DEFAULT_ATLAS_PATH) if DEFAULT_ATLAS_PATH.exists() else None
        backend.connect(); cameras.connect(); policy.reset(); pre.reset(); post.reset()
        last = None; keeper = RateKeeper(cfg.teleop.control_hz, enabled=True); started = time.perf_counter()
        if not cameras.roles:
            raise RuntimeError("policy rollout requires cameras; assign the same roles used by the dataset")
        for index in range(a.steps):
            measured = np.asarray(backend.read_joints_deg(), float); arm = measured[:5]
            pose = ik.task_pose(arm).pose
            obs: dict[str, np.ndarray] = {"observation.state": np.r_[pose.as_array(), measured[5]].astype(np.float32)}
            for role in cameras.roles:
                frame = cameras.read_latest(role)
                if frame is None: raise RuntimeError(f"no fresh frame from {role}")
                obs[f"observation.images.{role}"] = crop_image(frame, rois.get(role))
            action = predict_action(obs, policy, device, pre, post, policy.config.use_amp, task=a.task)
            target_raw, gripper = apply_action(pose.as_array(), action.detach().cpu().numpy()[0], a.action_space)
            update = tracker.approach(SO100TaskPose.from_array(target_raw)); target = update.pose
            if atlas is not None:
                target = atlas.project(target).pose; tracker.pose = target
            seed = arm if last is None else last
            result = ik.solve(seed, target, rate_reference_deg=seed)
            command = np.r_[result.joints_deg, gripper]
            if result.achieved.position[2] < cfg.limits.pos_min_m[2]: command = np.r_[seed, measured[5]]
            clearance = getattr(backend, "command_robot_mesh_min_z_deg", None)
            if callable(clearance) and clearance(command)[0] < cfg.teleop.min_robot_mesh_z_m: command = np.r_[seed, measured[5]]
            backend.write_joints_deg(command); last = command[:5]
            print(f"{index + 1}/{a.steps} {time.perf_counter()-started:.1f}s", flush=True); keeper.wait()
    finally:
        try: cameras.disconnect()
        finally: backend.disconnect()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
