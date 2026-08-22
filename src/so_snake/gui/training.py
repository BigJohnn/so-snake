"""Background training / policy-rollout jobs owned by the GUI gateway.

The gateway never shells out through a user supplied string.  A run is made of
validated fields and an argv list, which also makes the command shown in the UI
an accurate audit trail rather than a second, hand-maintained description.
"""
from __future__ import annotations

import os
import json
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from ..data.export import validate_roi

DEFAULT_MODEL_ROOT = REPO_ROOT / "outputs"


class TrainingManager:
    """One cancellable training-or-rollout process, plus a safe model library."""

    def __init__(self, model_root: Path = DEFAULT_MODEL_ROOT) -> None:
        self.model_root = Path(model_root)
        self.model_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._status: dict[str, Any] = self._idle()

    @staticmethod
    def _idle() -> dict[str, Any]:
        return {"kind": "", "running": False, "phase": "idle", "name": "", "pid": 0,
                "started_at": 0.0, "ended_at": 0.0, "exit_code": None, "error": "",
                "command": [], "log": []}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {**self._status, "log": list(self._status["log"])}

    def _safe_name(self, value: str) -> str:
        name = str(value).strip()
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("model name must be a single non-empty directory name")
        if not all(c.isalnum() or c in "._-" for c in name):
            raise ValueError("model name may only use letters, digits, '.', '_' and '-'")
        return name

    def _unique_name(self, value: str) -> str:
        """Reserve an unambiguous, human-sortable output name without overwriting.

        LeRobot deliberately refuses an existing output directory.  Adding the
        timestamp here is more useful than surfacing that refusal after a
        multi-minute dataset upload, and the short suffix protects two clicks
        that land in the same second.
        """
        base = self._safe_name(value)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate = f"{base}-{stamp}"
        index = 2
        while (self.model_root / candidate).exists():
            candidate = f"{base}-{stamp}-{index}"
            index += 1
        return candidate

    def _under_root(self, path: Path) -> Path:
        target = path.resolve()
        try:
            target.relative_to(self.model_root.resolve())
        except ValueError as exc:
            raise ValueError("model path escapes the model root") from exc
        return target

    def models(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for directory in sorted(self.model_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not directory.is_dir():
                continue
            candidates = [directory / "checkpoints" / "last" / "pretrained_model", directory / "pretrained_model"]
            checkpoint = next((p for p in candidates if (p / "config.json").is_file()), None)
            size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            policy = ""
            observation: dict[str, Any] = {}
            if checkpoint:
                try:
                    policy = str(json.loads((checkpoint / "config.json").read_text()).get("type", ""))
                except (OSError, json.JSONDecodeError):
                    pass
            profile = directory / "so_snake_observation.json"
            if profile.is_file():
                try:
                    observation = json.loads(profile.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
            records.append({"name": directory.name, "path": str(directory), "modified": directory.stat().st_mtime,
                            "size_bytes": size, "checkpoint": str(checkpoint) if checkpoint else "",
                            "ready": checkpoint is not None, "policy": policy,
                            "roi": observation.get("roi", {})})
        return {"root": str(self.model_root), "models": records}

    def _start(self, *, kind: str, name: str, argv: list[str], env: dict[str, str] | None = None,
               output: Path | None = None, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError(f"{self._status['kind']} is already running")
            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)
            process = subprocess.Popen(argv, cwd=REPO_ROOT, env=merged_env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            self._process = process
            self._status = {"kind": kind, "running": True, "phase": "running", "name": name,
                            "pid": process.pid, "started_at": time.time(), "ended_at": 0.0,
                            "exit_code": None, "error": "", "command": argv, "log": deque(maxlen=240)}
            self._status["output"] = str(output or "")
            self._status["observation"] = observation or {}
            threading.Thread(target=self._collect, args=(process,), daemon=True, name=f"so-snake-{kind}").start()
            return self.status()

    def _collect(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            with self._lock:
                self._status["log"].append(line.rstrip())
        code = process.wait()
        with self._lock:
            if self._process is process:
                self._process = None
                self._status.update({"running": False, "ended_at": time.time(), "exit_code": code,
                                     "phase": "done" if code == 0 else "failed"})
                if code != 0:
                    self._status["error"] = f"process exited with status {code}"
                elif self._status.get("output") and self._status.get("observation"):
                    profile = Path(str(self._status["output"])) / "so_snake_observation.json"
                    try:
                        profile.write_text(
                            json.dumps(self._status["observation"], indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    except OSError as exc:
                        self._status.update({"phase": "failed", "exit_code": 1,
                                             "error": f"could not write observation profile: {exc}"})

    def start_train(self, *, dataset: Path, policy: str, name: str, device: str, steps: int, batch_size: int,
                    base_model: str = "", wandb: dict[str, Any] | None = None,
                    autodl: dict[str, Any] | None = None, pi_mode: str = "expert") -> dict[str, Any]:
        if policy not in {"act", "pi05"}:
            raise ValueError("policy must be act or pi05")
        if policy == "pi05" and not base_model.strip():
            raise ValueError("pi0.5 training requires a pretrained base model")
        if policy == "pi05" and device.strip().lower() == "mps":
            raise ValueError("pi0.5 training is not supported on MPS; use an AutoDL CUDA GPU")
        if policy == "pi05" and pi_mode not in {"expert", "full"}:
            raise ValueError("pi0.5 mode must be expert or full")
        if not dataset.is_dir() or not (dataset / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"not a LeRobot dataset: {dataset}")
        name = self._unique_name(name)
        output = self._under_root(self.model_root / name)
        if steps < 1 or steps > 10_000_000 or batch_size < 1 or batch_size > 4096:
            raise ValueError("steps or batch size is outside the safe range")
        # `repo_id` is read from the export manifest by lerobot as a normal dataset ID;
        # the root pins it to the local bytes selected in the GUI.
        import json
        info = json.loads((dataset / "meta" / "info.json").read_text())
        repo_id = str(info.get("repo_id", ""))
        if not repo_id:
            raise ValueError("dataset metadata has no repo_id")
        observation = self._observation_profile(dataset)
        argv = [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--dataset.repo_id={repo_id}",
                f"--dataset.root={dataset}", f"--policy.type={policy}", f"--policy.device={device}",
                "--policy.push_to_hub=false", f"--output_dir={output}", f"--steps={steps}", f"--batch_size={batch_size}"]
        if base_model.strip():
            argv.append(f"--policy.pretrained_path={base_model.strip()}")
        if policy == "pi05":
            # This exporter predates pi05's q01/q99 statistics.  Mean/std is
            # the documented compatible path, while checkpointing + bf16 is
            # the safe CUDA baseline before the operator picks a larger GPU.
            argv.extend([
                '--policy.normalization_mapping={"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}',
                "--policy.gradient_checkpointing=true",
                "--policy.dtype=bfloat16",
            ])
            if pi_mode == "expert":
                argv.extend(["--policy.freeze_vision_encoder=true", "--policy.train_expert_only=true"])
        argv.extend(self._wandb_args(wandb))
        if autodl and bool(autodl.get("enabled")):
            return self._start_autodl(dataset, output, name, argv, autodl, observation)
        return self._start(kind="train", name=name, argv=argv, env={"HF_LEROBOT_HOME": str(dataset.parent)},
                           output=output, observation=observation)

    @staticmethod
    def _observation_profile(dataset: Path) -> dict[str, Any]:
        """Carry the exported visual contract into the model directory.

        Training reads already-cropped dataset images; the sidecar makes that
        otherwise invisible contract available to rollout, which receives raw
        camera frames and must crop them identically before inference.
        """
        manifest = dataset / "export.json"
        raw: dict[str, Any] = {}
        if manifest.is_file():
            try:
                raw = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"could not read dataset export manifest: {exc}") from exc
        roi = raw.get("roi", {})
        if not isinstance(roi, dict):
            raise ValueError("dataset export manifest has an invalid roi")
        return {"version": 1, "dataset": str(dataset), "roi": roi,
                "resolution": raw.get("resolution", []), "cameras": raw.get("cameras", [])}

    @staticmethod
    def _wandb_args(wandb: dict[str, Any] | None) -> list[str]:
        if not wandb or not bool(wandb.get("enabled")):
            return []
        project = str(wandb.get("project", "")).strip()
        if not project:
            raise ValueError("W&B project is required when W&B is enabled")
        result = ["--wandb.enable=true", f"--wandb.project={project}"]
        for key in ("entity", "notes", "mode"):
            value = str(wandb.get(key, "")).strip()
            if value:
                result.append(f"--wandb.{key}={value}")
        return result

    @staticmethod
    def _remote_value(value: Any, label: str, pattern: str) -> str:
        text = str(value or "").strip()
        if not text or not re.fullmatch(pattern, text):
            raise ValueError(f"invalid AutoDL {label}")
        return text

    def _start_autodl(self, dataset: Path, output: Path, name: str, train_argv: list[str], profile: dict[str, Any],
                      observation: dict[str, Any]) -> dict[str, Any]:
        """Sync one immutable dataset, run remotely, then retrieve its output.

        AutoDL exposes standard SSH; using it keeps account/API credentials out
        of the GUI and works with user-provided SSH keys or ssh-agent.  All
        remote strings are allowlisted before they reach rsync/ssh.
        """
        host = self._remote_value(profile.get("host"), "host", r"[A-Za-z0-9._-]+")
        user = self._remote_value(profile.get("user"), "user", r"[A-Za-z0-9_-]+")
        port = self._remote_value(profile.get("port", "22"), "port", r"[0-9]{1,5}")
        root = self._remote_value(profile.get("root", "/root/autodl-tmp"), "workspace", r"/[A-Za-z0-9._/-]+")
        python = self._remote_value(profile.get("python", "python"), "Python", r"[A-Za-z0-9._/-]+")
        argv = [sys.executable, str(REPO_ROOT / "scripts" / "autodl_train.py"), "--dataset", str(dataset),
                "--local-output", str(output), "--host", host, "--user", user, "--port", port,
                "--remote-root", root, "--remote-python", python, "--run-name", name, "--", *train_argv]
        return self._start(kind="autodl", name=name, argv=argv, output=output, observation=observation)

    def start_rollout(self, *, checkpoint: Path, task: str, argv_tail: list[str]) -> dict[str, Any]:
        checkpoint = Path(checkpoint).resolve()
        if not (checkpoint / "config.json").is_file() or not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError("checkpoint must contain config.json and model.safetensors")
        self._under_root(checkpoint)
        run_root = next((parent for parent in checkpoint.parents if parent.parent == self.model_root), None)
        profile = run_root / "so_snake_observation.json" if run_root else Path()
        if profile.is_file():
            try:
                observation = json.loads(profile.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"could not read model observation profile: {exc}") from exc
            roi = observation.get("roi", {})
            if not isinstance(roi, dict):
                raise ValueError("model observation profile has an invalid roi")
            for role, region in roi.items():
                if not isinstance(role, str):
                    raise ValueError("model observation profile has an invalid roi")
                try:
                    values = validate_roi(region, label=f"ROI for {role}")
                except ValueError as exc:
                    raise ValueError("model observation profile has an invalid roi") from exc
                argv_tail.extend(["--roi", f"{role}=" + ",".join(str(value) for value in values)])
        argv = [sys.executable, str(REPO_ROOT / "scripts" / "rollout_policy.py"), "--checkpoint", str(checkpoint),
                "--task", task, *argv_tail]
        return self._start(kind="rollout", name=checkpoint.parent.parent.parent.name, argv=argv)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.status()
            # The rollout owns a robot. SIGTERM gives it a chance to execute its
            # finally block and release torque; a hard kill follows only if needed.
            os.killpg(process.pid, signal.SIGTERM)
            self._status["phase"] = "stopping"
        return self.status()

    def delete_models(self, names: list[str]) -> dict[str, Any]:
        import shutil
        if not isinstance(names, list) or not names:
            raise ValueError("select at least one model to delete")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("stop the active train/rollout before deleting models")
        deleted: list[str] = []
        for raw in names:
            path = self._under_root(self.model_root / self._safe_name(raw))
            if path.is_dir():
                shutil.rmtree(path)
                deleted.append(path.name)
        return {"deleted": deleted}
