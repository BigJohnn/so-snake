"""The HTTP gateway: a JSON API over `SessionManager`, plus the built frontend.

Standard-library `http.server`, deliberately. The API is a dozen endpoints
serving one operator on one machine, and a web framework would be the heaviest
dependency in a repository whose base install is `numpy` -- for a page that
exists to make the offline gates easier to run.

Design rules that are worth keeping if this grows:

  * **the gateway holds no robot state.** Everything lives in `SessionManager`,
    which is also what the CLI scripts drive. A handler that started keeping its
    own idea of whether the arm is moving would be a second source of truth for
    the one fact that must not have two.
  * **every mutation is a POST that returns the new snapshot.** The UI never has
    to guess what its click did, and never has to poll to find out.
  * **failures are 4xx/5xx with the message, not a 200 with a flag.** A refused
    replay is refused loudly.
"""

from __future__ import annotations

import json
import mimetypes
import os
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..config import REPO_ROOT, SoSnakeConfig
from ..data import DEFAULT_EPISODE_ROOT, ReplayConfig
from ..m0_perception import CAMERA_ROLES, CameraSpec, list_devices
from ..rig import RigSpec, availability, mujoco_import_error
from .preview import encode_png, ensure_headless_gl, placeholder_png
from .roadmap import roadmap_payload
from .session import SessionManager

DEFAULT_FRONTEND_DIST = REPO_ROOT / "tools" / "gui" / "frontend" / "dist"

# How many points of an episode's series the detail view gets. The plots are a
# few hundred pixels wide, so sending 50 000 rows would be moving data nobody
# can see; the decimation is stride-based and keeps the endpoints.
DETAIL_SERIES_POINTS = 1200

PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 960


class Gateway:
    """Everything the handlers need, in one object."""

    def __init__(
        self,
        config: SoSnakeConfig | None = None,
        episode_root: Path = DEFAULT_EPISODE_ROOT,
        frontend_dist: Path = DEFAULT_FRONTEND_DIST,
    ) -> None:
        self.config = config or SoSnakeConfig()
        self.session = SessionManager(self.config, episode_root)
        self.frontend_dist = Path(frontend_dist)
        self.lock = threading.Lock()

    # -------------------------------------------------------------- payloads

    def config_payload(self) -> dict[str, Any]:
        """The tuning the UI needs to draw meaningful axes and limits.

        Sent rather than duplicated in TypeScript: the workspace box and the
        joint limits change during tuning, and a UI drawing last month's box
        around this month's arm is worse than one drawing no box.
        """
        arm, limits, teleop = self.config.arm, self.config.limits, self.config.teleop
        return {
            "arm": {
                "joint_names": list(arm.joint_names),
                "gripper_joint": "gripper",
                "joint_limits_deg": {k: list(v) for k, v in arm.joint_limits_deg.items()},
                "urdf_path": str(arm.urdf_path),
            },
            "limits": {
                "pos_min_m": list(limits.pos_min_m),
                "pos_max_m": list(limits.pos_max_m),
                "pitch_min_deg": float(np.degrees(limits.pitch_min_rad)),
                "pitch_max_deg": float(np.degrees(limits.pitch_max_rad)),
                "max_step_pos_m": limits.max_step_pos_m,
            },
            "teleop": {
                "control_hz": teleop.control_hz,
                "home_joints_deg": list(teleop.home_joints_deg),
                "max_joint_step_deg": teleop.max_joint_step_deg,
                "min_robot_mesh_z_m": teleop.min_robot_mesh_z_m,
                "gripper_open_deg": teleop.gripper_open_deg,
                "gripper_closed_deg": teleop.gripper_closed_deg,
            },
            "episode_root": str(self.session.store.root),
            "availability": availability(),
            "cameras": list(CAMERA_ROLES),
            "gl_backend": os.environ.get("MUJOCO_GL", ""),
        }

    def cameras_payload(self) -> dict[str, Any]:
        """Enumerate the cameras attached to this machine.

        Deliberately not part of `/api/config`, which the UI polls: listing
        means opening each device and pulling a frame, which takes seconds and
        competes with a session that already has one of them open. The UI asks
        for this on demand, and refuses to ask while the arm is busy.
        """
        if self.session.busy:
            raise RuntimeError(
                "cannot scan for cameras while a session is running -- "
                "the scan would open devices the session is using"
            )
        return {"devices": list_devices(), "roles": list(CAMERA_ROLES)}

    def episodes_payload(self) -> dict[str, Any]:
        return {
            "episodes": [
                {**meta.to_json(), "config": None}  # the config snapshot is per-episode detail
                for meta in self.session.store.list_meta()
            ]
        }

    def episode_detail(self, episode_id: str) -> dict[str, Any]:
        """One episode's metadata plus a decimated view of its series."""
        episode = self.session.store.load(episode_id)
        n = len(episode.commanded_joints_deg)
        stride = max(1, n // DETAIL_SERIES_POINTS)
        keep = np.arange(0, n, stride)

        def sub(name: str) -> np.ndarray:
            return np.asarray(episode.column(name))[keep]

        target = sub("action.task.target")
        achieved = sub("observation.state.task_pose")
        return {
            "meta": episode.meta.to_json(),
            "size_bytes": self.session.store.disk_usage_bytes(episode_id),
            "columns": sorted(episode.frames),
            "series": {
                "stride": int(stride),
                "t": sub("t").round(3).tolist(),
                "task_target": target.round(5).tolist(),
                "achieved_task_pose": achieved.round(5).tolist(),
                "commanded_joints_deg": sub("action.joint.commanded_deg").round(3).tolist(),
                "measured_joints_deg": sub("observation.state.joints_deg").round(3).tolist(),
                "gripper_deg": sub("action.task.gripper_deg").round(3).tolist(),
                "clutch": sub("diagnostics.clutch_engaged").astype(bool).tolist(),
                "pos_err_mm": (sub("diagnostics.ik_position_error_m") * 1000.0).round(4).tolist(),
                "loop_hz": np.where(
                    sub("diagnostics.loop_dt_s") > 0,
                    1.0 / np.maximum(sub("diagnostics.loop_dt_s"), 1e-9),
                    0.0,
                ).round(1).tolist(),
                "workspace_clamped": sub("diagnostics.workspace_clamped").astype(bool).tolist(),
                "atlas_pitch_clamped": sub("diagnostics.atlas_pitch_clamped").astype(bool).tolist(),
                "joint_rate_clamped": sub("diagnostics.joint_rate_clamped").astype(bool).tolist(),
                "command_safety_held": sub("diagnostics.command_safety_held").astype(bool).tolist(),
            },
        }


# ------------------------------------------------------------------ decoding


def cameras_from_body(body: dict[str, Any]) -> tuple[CameraSpec, ...]:
    """Read the role -> device assignments out of a request body.

    Shaped `{"cameras": {"wrist": 3, "third_person": "/dev/video0"}}`. A role
    mapped to an empty string or null means "not this session" and is dropped
    rather than rejected, so the UI can clear a picker without having to send a
    different shape. `RigSpec.validate` rejects unknown roles.
    """
    raw = body.get("cameras") or {}
    if not isinstance(raw, dict):
        raise ValueError("cameras must be an object of role -> device")

    specs: list[CameraSpec] = []
    for role, device in raw.items():
        if device is None or (isinstance(device, str) and not device.strip()):
            continue
        # An index arrives as a JSON number, a path as a string; a numeric
        # string is an index typed into a text field, which is the common case
        # on macOS where devices have no paths.
        if isinstance(device, str) and device.strip().lstrip("-").isdigit():
            device = int(device.strip())
        elif isinstance(device, bool):  # bool is an int subclass; not a device
            raise ValueError(f"camera {role!r} has an invalid device")
        elif not isinstance(device, (int, str)):
            raise ValueError(f"camera {role!r} has an invalid device")
        specs.append(CameraSpec(role=str(role), index_or_path=device))
    return tuple(specs)


def spec_from_body(body: dict[str, Any], *, default_backend: str = "mock") -> RigSpec:
    """Build a `RigSpec` from a request body, taking nothing on trust.

    `RigSpec.validate` catches the enums; the clamps below catch the numbers,
    because `max_relative_target_deg` is a hardware safety limit and a UI bug
    that posted 500 must not become the arm's per-step budget.
    """
    spec = RigSpec(
        cameras=cameras_from_body(body),
        backend=str(body.get("backend", default_backend)),
        source=str(body.get("source", "scripted")),
        port=str(body.get("port", "")),
        robot_id=str(body.get("robot_id", "so_snake")),
        max_relative_target_deg=float(
            np.clip(float(body.get("max_relative_target_deg", 5.0)), 0.5, 20.0)
        ),
        gripper_speed_mult=float(np.clip(float(body.get("gripper_speed_mult", 3.0)), 1.0, 6.0)),
        device_id=body.get("device_id"),
        scripted_amplitude=float(np.clip(float(body.get("amplitude", 0.2)), 0.0, 1.0)),
        scripted_rotation_amplitude_rad=float(
            np.clip(float(body.get("rotation_amplitude", 0.10)), 0.0, 1.0)
        ),
        scripted_loop=True,
    )
    spec.validate()
    return spec


def replay_from_body(body: dict[str, Any]) -> ReplayConfig:
    return ReplayConfig(
        mode=str(body.get("mode", "joint")),
        speed=float(np.clip(float(body.get("speed", 1.0)), 0.05, 4.0)),
        realtime=bool(body.get("realtime", True)),
        check_clearance=bool(body.get("check_clearance", True)),
    )


# ------------------------------------------------------------------- handler


class GuiHandler(BaseHTTPRequestHandler):
    server: "GuiServer"
    server_version = "so-snake-gui"

    # The default logs one line per request to stderr, which at 10 Hz polling
    # buries anything worth reading.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        return

    # ------------------------------------------------------------- plumbing

    @property
    def gateway(self) -> Gateway:
        return self.server.gateway

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, allow_nan=False, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed JSON body: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def _guard(self, action: Callable[[], Any]) -> None:
        """Run an action, mapping its failures onto status codes.

        The three cases are genuinely different and the UI shows them
        differently: a bad request is the caller's fault, a conflict means the
        arm is busy and the operator should stop what is running, and anything
        else is a bug worth surfacing rather than swallowing.
        """
        try:
            self._json(HTTPStatus.OK, action())
        except (ValueError, KeyError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, str(exc))
        except RuntimeError as exc:
            self._error(HTTPStatus.CONFLICT, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ GET

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if path == "/api/health":
            self._json(HTTPStatus.OK, {"ok": True, "mode": self.gateway.session.mode})
        elif path == "/api/config":
            self._guard(self.gateway.config_payload)
        elif path == "/api/roadmap":
            self._guard(roadmap_payload)
        elif path == "/api/snapshot":
            self._guard(self.gateway.session.status)
        elif path == "/api/series":
            limit = _int_param(query, "limit", 600)
            self._guard(lambda: {"series": self.gateway.session.series(limit)})
        elif path == "/api/episodes":
            self._guard(self.gateway.episodes_payload)
        elif path == "/api/episode":
            episode_id = _str_param(query, "id")
            self._guard(lambda: self.gateway.episode_detail(episode_id))
        elif path == "/api/cameras":
            self._guard(self.gateway.cameras_payload)
        elif path == "/api/episode/video":
            self._serve_episode_video(_str_param(query, "id"), _str_param(query, "camera"))
        elif path == "/api/preview.png":
            self._serve_preview(query)
        elif path.startswith("/api/"):
            self._error(HTTPStatus.NOT_FOUND, f"no such endpoint: {path}")
        else:
            self._serve_frontend(path)

    def _serve_episode_video(self, episode_id: str, camera: str) -> None:
        """Stream one episode's camera file, with byte ranges.

        Range support is not optional here: a `<video>` element that cannot ask
        for a byte range cannot seek, and an episode video that only plays from
        the start is useless for reviewing the moment a take went wrong.
        `SimpleHTTPRequestHandler` does not implement it, so it is done here.
        """
        if camera not in CAMERA_ROLES:
            self._error(HTTPStatus.BAD_REQUEST, f"unknown camera: {camera}")
            return
        try:
            directory = self.gateway.session.store.path_of(episode_id)
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        path = directory / f"{camera}.mp4"
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"episode {episode_id} has no {camera} video")
            return

        size = path.stat().st_size
        start, end = 0, size - 1
        partial = False
        header = self.headers.get("Range", "")
        if header.startswith("bytes="):
            first, _, last = header[len("bytes="):].partition("-")
            try:
                if first:
                    start = int(first)
                    end = int(last) if last else size - 1
                else:  # a suffix range: the last N bytes
                    start = max(0, size - int(last))
                partial = True
            except ValueError:
                partial = False
                start, end = 0, size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _serve_preview(self, query: dict[str, list[str]]) -> None:
        camera = _str_param(query, "camera", "third_person")
        if camera not in ("third_person", "wrist"):
            self._error(HTTPStatus.BAD_REQUEST, f"unknown camera: {camera}")
            return
        width = min(_int_param(query, "w", 640), PREVIEW_MAX_WIDTH)
        height = min(_int_param(query, "h", 480), PREVIEW_MAX_HEIGHT)
        try:
            frame = self.gateway.session.preview_frame(camera, width, height)
            png = encode_png(frame) if frame is not None else placeholder_png(width, height)
        except Exception as exc:  # noqa: BLE001 - a broken GL stack must not 500 the page
            self.gateway.session.log("warn", f"preview render failed: {type(exc).__name__}: {exc}")
            png = placeholder_png(width, height)

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(png)

    def _serve_frontend(self, path: str) -> None:
        """Serve the built single-page app, falling back to `index.html`.

        Any path that is not a file is the SPA's own route, so it gets the
        shell. Requests are resolved and checked against the dist root, since
        this server is reachable from the network when bound to 0.0.0.0.
        """
        dist = self.gateway.frontend_dist
        index = dist / "index.html"
        if not index.is_file():
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": "frontend not built",
                    "hint": f"cd {dist.parent} && npm install && npm run build",
                },
            )
            return

        candidate = (dist / path.lstrip("/")).resolve() if path not in ("", "/") else index
        try:
            inside = candidate.is_relative_to(dist.resolve())
        except (OSError, ValueError):
            inside = False
        target = candidate if inside and candidate.is_file() else index

        data = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Vite fingerprints its assets, so they are immutable; the shell is not.
        self.send_header(
            "Cache-Control",
            "no-store" if target == index else "public, max-age=31536000, immutable",
        )
        self.end_headers()
        self.wfile.write(data)

    # ----------------------------------------------------------------- POST

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        path = urlparse(self.path).path
        session = self.gateway.session
        try:
            body = self._body()
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        if path == "/api/session/start":
            self._guard(lambda: session.start_session(spec_from_body(body)))
        elif path == "/api/session/stop":
            self._guard(session.stop)
        elif path == "/api/session/home":
            self._guard(lambda: session.start_homing(spec_from_body(body)))
        elif path == "/api/record/start":
            self._guard(
                lambda: session.start_recording(
                    name=str(body.get("name", "")),
                    task=str(body.get("task", "")),
                    notes=str(body.get("notes", "")),
                )
            )
        elif path == "/api/record/stop":
            self._guard(lambda: session.stop_recording(keep=bool(body.get("keep", True))))
        elif path == "/api/replay/start":
            self._guard(
                lambda: session.start_replay(
                    str(body.get("episode_id", "")),
                    spec_from_body(body),
                    replay_from_body(body),
                )
            )
        elif path == "/api/replay/stop":
            self._guard(session.stop)
        elif path == "/api/episode/annotate":
            self._guard(
                lambda: self.gateway.session.store.annotate(
                    str(body.get("id", "")),
                    name=body.get("name"),
                    task=body.get("task"),
                    notes=body.get("notes"),
                ).to_json()
            )
        elif path == "/api/episode/delete":
            self._guard(lambda: {"deleted": session.store.delete(str(body.get("id", "")))})
        else:
            self._error(HTTPStatus.NOT_FOUND, f"no such endpoint: {path}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _str_param(query: dict[str, list[str]], key: str, default: str = "") -> str:
    values = query.get(key, [])
    return values[0] if values else default


def _int_param(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(_str_param(query, key, str(default)))
    except ValueError:
        return default


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], gateway: Gateway) -> None:
        super().__init__(address, GuiHandler)
        self.gateway = gateway


def serve(
    host: str = "127.0.0.1",
    port: int = 8770,
    *,
    episode_root: Path = DEFAULT_EPISODE_ROOT,
    frontend_dist: Path = DEFAULT_FRONTEND_DIST,
) -> None:
    # Before anything can import mujoco: it reads MUJOCO_GL at import time, and
    # a bad choice makes the import itself raise rather than degrading.
    gl_backend, gl_why = ensure_headless_gl()

    gateway = Gateway(episode_root=episode_root, frontend_dist=frontend_dist)
    server = GuiServer((host, port), gateway)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    # Flushed, because this is diagnostics: run under a supervisor or piped to a
    # log, a buffered banner does not appear until the server exits, which is
    # exactly when nobody needs to be told which GL backend it chose.
    print(f"so-snake GUI on http://{shown}:{port}")
    print(f"  episodes  {gateway.session.store.root}")
    print(f"  MUJOCO_GL {gl_backend or '(none)'}  — {gl_why}")
    mujoco_error = mujoco_import_error()
    if mujoco_error:
        print(f"  WARNING   the mujoco backend is unavailable: {mujoco_error}")
    print(
        f"  frontend  {frontend_dist}"
        + ("" if (frontend_dist / "index.html").is_file() else "  (not built — API only)"),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down ...")
    finally:
        gateway.session.shutdown()
        server.server_close()
