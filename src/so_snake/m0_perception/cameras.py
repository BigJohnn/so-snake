"""The real wrist and third-person cameras.

Capture goes through lerobot's `OpenCVCamera` rather than through `cv2` here.
That is reuse, not indirection: it already owns a per-camera capture thread, the
warmup, the colour-mode conversion and the rotation, and this repository's
standing relationship with lerobot is to depend on it and not reimplement it.
It also costs nothing to install -- `opencv-python-headless` is an unconditional
dependency of lerobot, so any machine that can drive the real arm already has
OpenCV on disk. Cameras therefore live in the `.[teleop]` extra, and every
import of lerobot below is deferred so that the offline gates, the sim and the
scripted viewer keep running on an install that has none of it.

Two things learned from the hardware on this bench, both of which shape the API:

**The requested resolution is a request.** `OpenCVCamera._configure_capture_settings`
sets width/height and *raises* if the device did not adopt them. Of the two
identical DECXIN UVC cameras here, one accepts 640x480 and the other ignores it
and keeps streaming 1920x1080. So no resolution is requested at all: each camera
runs at whatever it natively produces, and `frame_for_preview` scales the result.
Native capture is what a recorded episode wants anyway -- downscaling for a
browser pane is not a decision to bake into the data.

**A camera index cannot be named, so it is shown instead.** On macOS the OpenCV
index space does not match `system_profiler`'s order, and does not match
ffmpeg's AVFoundation listing either -- all three enumerated the same four
cameras in different orders on this bench. Any label derived from one of them
is a guess, and a wrong guess here is silent: the built-in webcam and a USB
camera are both 1080p and both deliver frames, so an episode recorded from the
wrong one looks fine until somebody watches it. `list_devices` therefore returns
a thumbnail of what each index actually sees and no name at all, and the
operator assigns roles by looking. That is also the only thing that can separate
two identical cameras, which is the case on this bench.

Reads are `read_latest`, never `read`: the control loop must never block on a
USB frame. A stale frame is the correct answer to "what does the camera see" if
the alternative is a control step that missed its deadline.
"""

from __future__ import annotations

import base64
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# The roles the rest of the system knows about. These are the MuJoCo camera
# names too, so a preview pane asks for the same string whether the frame comes
# from a simulated camera or a real one.
CAMERA_ROLES: tuple[str, ...] = ("third_person", "wrist")

# A preview frame older than this is not shown. At 30 fps a camera that is
# alive is at most ~33 ms stale; half a second means the capture thread has
# stalled or the device has gone away, and a frozen image with no indication
# that it is frozen is worse than a blank pane.
MAX_FRAME_AGE_MS = 500


@dataclass(frozen=True)
class CameraSpec:
    """One camera, and the role it plays."""

    role: str
    index_or_path: int | str

    def validate(self) -> None:
        if self.role not in CAMERA_ROLES:
            raise ValueError(f"camera role must be one of {CAMERA_ROLES}, got {self.role!r}")
        if isinstance(self.index_or_path, str) and not self.index_or_path.strip():
            raise ValueError(f"camera {self.role!r} has an empty device")


def cameras_import_error() -> str:
    """Why cameras cannot be used here, or "" if they can.

    Imports for real rather than checking `find_spec`, for the same reason
    `mujoco_import_error` does: an installed package that raises on import is
    not a usable one, and offering the operator a camera that 500s when they
    select it is worse than saying up front that it is unavailable.
    """
    try:
        from lerobot.cameras.opencv import OpenCVCamera  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - the reason is the payload
        return f"{type(exc).__name__}: {exc}"
    return ""


# --------------------------------------------------------------------- naming


def _thumbnail(frame_bgr: np.ndarray, width: int = 192) -> str:
    """A small JPEG of `frame_bgr` as a data URI, for identifying a device by eye.

    Takes BGR because that is what a bare `cv2.VideoCapture.read` returns, and
    `cv2.imencode` expects the same -- so the frame goes in untouched. This is
    the one place in this module that handles BGR: `CameraRig` reads through
    lerobot with `ColorMode.RGB`, so everything downstream of it is RGB.
    Converting here as well swaps red and blue, which is a wrong-looking
    thumbnail rather than an error, and the thumbnail is what the operator is
    identifying a camera by.
    """
    import cv2

    height = max(1, int(round(frame_bgr.shape[0] * width / max(1, frame_bgr.shape[1]))))
    small = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


# Below this, a frame carries too little detail to tell one camera from another
# by eye. That is a statement about *identifying* the device during a scan, and
# emphatically not a statement that anything is wrong with it.
#
# The number is the variance of the Laplacian, the standard focus measure.
# Across repeated scans of the five devices on this bench:
#
#     third-person DECXIN   120-128   aimed at the workspace, sharp
#     OBS virtual camera         74   a rendered logo
#     built-in FaceTime       10-15   in focus, pointed at a plain room
#     wrist DECXIN           2.2-4.3  near-focused, nothing in front of it
#     iPhone (locked)           0.0   a black frame forever
#
# **The wrist camera reading low is correct behaviour.** Its lens is focused at
# roughly gripper distance, which is the only distance at which its picture is
# worth anything -- the policy needs to see the object as the jaws close on it,
# not the far side of the room. With nothing in front of the gripper it images a
# blurred background, and that is the lens doing its job. An earlier version of
# this file called that "badly out of focus" and told the operator to turn the
# focus ring, which would have broken a correctly set up camera.
#
# So this measure gates nothing. It never has: no code path refuses to open,
# record, or export on account of it. What it does is answer the question the
# scan exists to answer -- *which thumbnail is which camera* -- because a
# low-detail thumbnail in a row of five reads as an empty slot, and the operator
# concludes the camera did not show up when it is right there.
#
# The measure is scene-dependent rather than purely optical (a sharp camera
# aimed at a blank wall scores low too), so no threshold separates the
# populations perfectly. It does not need to: a false positive costs one
# unnecessary line of hint text.
LOW_DETAIL_LAPLACIAN_VAR = 6.0

# Below this the frame is a flat field -- a lens cap, a covered camera, or a
# device that streams a black frame forever, which is what a locked iPhone
# offering itself as a Continuity Camera does. Unlike low detail, this one
# really is "no picture", though it still blocks nothing.
BLANK_CONTRAST_STD = 6.0


def _sharpness(frame_bgr: np.ndarray) -> tuple[float, float]:
    """`(laplacian variance, contrast std)` for a captured frame.

    Both are computed on the full-resolution frame rather than the thumbnail:
    downscaling is itself a low-pass filter, so a thumbnail's Laplacian would
    report every camera as softer than it is and the threshold would stop
    separating them.
    """
    import cv2

    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var()), float(grey.std())


def _picture_quality(frame_bgr: np.ndarray) -> dict[str, Any]:
    """Measures of the frame, and a note if it is hard to identify by eye.

    The note is an aid to picking the right device out of a row of thumbnails.
    It is never a verdict on the camera: a scan cannot know which role a device
    is about to be given, and for the wrist role a low-detail picture at rest is
    the expected and correct thing to see.
    """
    sharpness, contrast = _sharpness(frame_bgr)
    if contrast < BLANK_CONTRAST_STD:
        note = (
            "no picture -- a flat frame with no contrast at all. The lens is "
            "covered, or the device streams a fixed frame (a locked iPhone "
            "offering itself as a Continuity Camera does this)"
        )
    elif sharpness < LOW_DETAIL_LAPLACIAN_VAR:
        note = (
            "little detail at this distance, so it is hard to tell apart from "
            "another camera by its picture. This is normal and correct for a "
            "wrist camera at rest -- it is focused at gripper distance and only "
            "needs to be sharp there. To identify it, hold something a few "
            "centimetres in front of the lens and scan again"
        )
    else:
        note = ""
    return {
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "note": note,
    }


def _linux_camera_names() -> dict[int, str]:
    """`/dev/videoN` -> friendly name, from sysfs. Best effort."""
    from pathlib import Path

    names: dict[int, str] = {}
    for node in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            index = int(node.name.removeprefix("video"))
            names[index] = (node / "name").read_text().strip()
        except (OSError, ValueError):
            continue
    return names


def _linux_stable_paths() -> dict[int, str]:
    """`/dev/videoN` -> its `/dev/v4l/by-id/...` symlink, where one exists.

    This is the identifier macOS does not have. udev builds the name from the
    device's USB vendor, product, serial *and* physical port path, so two
    cameras of the same model get different links, and a link keeps pointing at
    the same physical camera across reboots and across other devices coming and
    going -- which is exactly what a bare index does not do.

    OpenCV opens a path as readily as an index, and `CameraSpec.index_or_path`
    already carries either, so a rig configured this way survives a replug.
    Moving the camera to a different USB port does change the link, because the
    port is part of what makes it unique; that is the honest behaviour, since
    the operator has physically changed the setup.
    """
    from pathlib import Path

    by_id = Path("/dev/v4l/by-id")
    if not by_id.is_dir():
        return {}

    paths: dict[int, str] = {}
    for link in sorted(by_id.iterdir()):
        try:
            target = link.resolve()
            index = int(target.name.removeprefix("video"))
        except (OSError, ValueError):
            continue
        # A camera exposes several nodes (capture, metadata); the lowest is the
        # capture one, and it is the one OpenCV wants.
        if index not in paths:
            paths[index] = str(link)
    return paths


def scan_devices(max_index: int = 16, thumbnails: bool = True) -> dict[str, Any]:
    """Enumerate cameras and explain what happened to the rejected candidates.

    `isOpened()` is not the test -- on macOS a Continuity Camera opens happily
    and then never produces an image -- so each candidate must hand over a frame
    to be listed. That makes this slow (a second or so per device) and means it
    must not be called from a poll; the GUI asks for it on demand.

    **The thumbnail is the identification, not the name.** On macOS there is no
    dependable way to say what device an OpenCV index is. Three enumerations of
    the same cameras on this machine disagreed with each other:

        OpenCV index      0=DECXIN(workspace) 1=DECXIN 2=FaceTime 3=OBS virtual
        system_profiler   0=OBS 1=FaceTime 2=DECXIN 3=DECXIN
        ffmpeg avfoundation  0=OBS 1=FaceTime 2=DECXIN 3=DECXIN

    An earlier version of this function labelled devices by zipping
    `system_profiler`'s order onto the OpenCV index, and so confidently reported
    the built-in FaceTime camera as a DECXIN. Nothing downstream could catch
    that: both are 1080p and both deliver frames. The names are gone, and the
    operator picks by looking at what each device sees -- which is also the only
    thing that can tell two identical cameras apart.

    Linux keeps its names, because there they are not a guess: OpenCV index N
    opens `/dev/videoN`, and the name is read from that same node's sysfs entry.

    Opening a camera triggers the operating system's permission prompt on
    macOS. If nothing is listed on a machine that visibly has cameras, that
    permission is the first thing to check: a denied process sees no devices
    rather than an error.
    """
    diagnostics: dict[str, Any] = {
        "platform": platform.system(),
        "max_index": int(max_index),
        "attempted": 0,
        "opened": 0,
        "readable": 0,
        "failures": [],
        # Devices whose picture carries too little detail to identify them by.
        # They are still in `devices` and nothing refuses them; this exists so
        # the UI can point at the thumbnail that looks like an empty slot and
        # say which camera it is, instead of the operator concluding it did not
        # show up. A wrist camera at rest lands here by design.
        "hard_to_identify": [],
        "permission_hint": "",
    }

    error = cameras_import_error()
    if error:
        diagnostics["permission_hint"] = f"camera dependencies are unavailable: {error}"
        return {"devices": [], "diagnostics": diagnostics}

    import cv2

    is_linux = platform.system() == "Linux"
    linux_names = _linux_camera_names() if is_linux else {}
    linux_paths = _linux_stable_paths() if is_linux else {}

    devices: list[dict[str, Any]] = []
    for index in range(max_index):
        diagnostics["attempted"] += 1
        capture = cv2.VideoCapture(index)
        try:
            if not capture.isOpened():
                diagnostics["failures"].append({"index": index, "reason": "did not open"})
                continue
            diagnostics["opened"] += 1
            ok, frame = False, None
            # A few frames in: the first one out of a UVC camera is often the
            # sensor's warm-up, and a black or half-exposed thumbnail is no use
            # for telling one camera from another.
            for _ in range(5):
                ok, frame = capture.read()
            if not ok or frame is None:
                # Opens but does not stream. An iPhone offering itself as a
                # Continuity Camera does exactly this when it is locked.
                diagnostics["failures"].append({"index": index, "reason": "opened but produced no frame"})
                continue
            diagnostics["readable"] += 1
            height, width = frame.shape[:2]
            quality = _picture_quality(frame)
            if quality["note"]:
                diagnostics["hard_to_identify"].append(
                    {"index": index, "reason": quality["note"]}
                )
            # `device` is what gets stored in the rig and the episode: the
            # stable by-id path where the platform offers one, and the bare
            # index -- which is only meaningful until something is replugged --
            # where it does not.
            stable = linux_paths.get(index, "")
            devices.append(
                {
                    "index": index,
                    "device": stable or index,
                    "stable": bool(stable),
                    # "usb" only where it is known, never inferred. udev names a
                    # by-id link after the bus it found the device on, so on
                    # Linux this is read, not guessed. On macOS it stays unknown:
                    # AVFoundation does report a transport type ('usb ', 'bltn',
                    # 'virt'), but there is no way to say which OpenCV index each
                    # of those devices is, and a filter that hides the wrong
                    # camera is worse than no filter -- the operator would trust
                    # a list that quietly dropped the camera they wanted.
                    "bus": "usb" if Path(stable).name.startswith("usb-") else "",
                    "name": linux_names.get(index, ""),
                    "width": int(width),
                    "height": int(height),
                    "thumbnail": _thumbnail(frame) if thumbnails else "",
                    # `sharpness`/`contrast`/`note`. Advisory only -- nothing
                    # anywhere refuses a device on account of them, because a
                    # scan cannot know which role a camera is about to be given
                    # and a near-focused wrist camera is supposed to look flat.
                    **quality,
                }
            )
        finally:
            capture.release()

    if not devices and platform.system() == "Darwin":
        diagnostics["permission_hint"] = (
            "macOS did not allow this process to read any camera. Grant Camera "
            "permission to the terminal/app that starts scripts/run_gui.sh, then scan again."
        )
    return {"devices": devices, "diagnostics": diagnostics}


def list_devices(max_index: int = 16, thumbnails: bool = True) -> list[dict[str, Any]]:
    """Only the devices, for callers that do not need scan diagnostics."""
    return list(scan_devices(max_index=max_index, thumbnails=thumbnails)["devices"])


# ------------------------------------------------------------------ capture


def _open_lerobot_camera(spec: CameraSpec) -> Any:
    """Open one camera through lerobot and start its capture thread."""
    error = cameras_import_error()
    if error:
        raise RuntimeError(f"cameras need the .[teleop] extra -- {error}")

    from lerobot.cameras.configs import ColorMode
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    # No fps/width/height: see the module docstring. Whatever the device
    # natively streams is what gets captured.
    camera = OpenCVCamera(
        OpenCVCameraConfig(index_or_path=spec.index_or_path, color_mode=ColorMode.RGB)
    )
    camera.connect()
    return camera


class CameraRig:
    """The cameras for one session, keyed by role.

    Connecting is all-or-nothing on purpose. A rig that comes up with the wrist
    camera missing would record episodes that are silently short a view, and
    that is only discovered when someone tries to train on them.
    """

    def __init__(
        self,
        specs: tuple[CameraSpec, ...] = (),
        open_camera: Callable[[CameraSpec], Any] | None = None,
    ) -> None:
        for spec in specs:
            spec.validate()
        roles = [spec.role for spec in specs]
        if len(set(roles)) != len(roles):
            raise ValueError(f"two cameras claim the same role: {roles}")
        self._specs = tuple(specs)
        # The seam exists so the lifecycle -- and particularly the rollback when
        # the second of two cameras fails to open -- can be tested without two
        # USB cameras present. Leaking a device handle there would be found by
        # the next session failing to open it, which is a bad place to find it.
        self._open_camera = open_camera or _open_lerobot_camera
        self._cameras: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def specs(self) -> tuple[CameraSpec, ...]:
        return self._specs

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(spec.role for spec in self._specs)

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return bool(self._cameras)

    def connect(self) -> None:
        """Open every camera and start its capture thread."""
        if not self._specs:
            return

        opened: dict[str, Any] = {}
        try:
            for spec in self._specs:
                opened[spec.role] = self._open_camera(spec)
        except Exception:
            for camera in opened.values():
                try:
                    camera.disconnect()
                except Exception:  # noqa: BLE001 - already failing; keep closing
                    pass
            raise

        with self._lock:
            self._cameras = opened

    def disconnect(self) -> None:
        with self._lock:
            cameras, self._cameras = self._cameras, {}
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass

    def read_latest(self, role: str) -> np.ndarray | None:
        """The newest frame for `role`, or None if there is not a fresh one.

        Never blocks on the device: this is a peek at what the capture thread
        has already put down, so it is safe to call from the control loop. None
        covers every not-right-now case -- no such camera, nothing captured
        yet, capture stalled -- because none of them is a reason to interrupt
        whatever the caller is doing.
        """
        with self._lock:
            camera = self._cameras.get(role)
        if camera is None:
            return None
        try:
            return camera.read_latest(max_age_ms=MAX_FRAME_AGE_MS)
        except Exception:  # noqa: BLE001 - stale, unstarted or gone: all "no frame"
            return None

    def status(self) -> dict[str, Any]:
        with self._lock:
            connected = set(self._cameras)
        return {
            "roles": list(self.roles),
            "devices": {spec.role: spec.index_or_path for spec in self._specs},
            "connected": sorted(connected),
        }


def frame_for_preview(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit a captured frame into `width` x `height` for a browser pane.

    Letterboxed rather than stretched, because the preview is what the operator
    judges the camera's aim by and a squashed image misleads about framing. The
    resampling uses OpenCV, which is present whenever a frame exists to resize.
    """
    import cv2

    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    source_h, source_w = image.shape[:2]
    if source_h == 0 or source_w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = min(width / source_w, height / source_h)
    new_w, new_h = max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale)))
    # INTER_AREA is the right filter for shrinking; it averages the pixels that
    # collapse together instead of point-sampling one of them.
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image[:, :, :3], (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas
