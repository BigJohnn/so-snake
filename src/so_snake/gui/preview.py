"""Camera preview: MuJoCo frames as PNG, with no image library.

A PNG encoder is about thirty lines on top of `zlib`, which is in the standard
library, and that is cheaper than making `Pillow` or `opencv` a dependency of a
repository whose entire base install is `numpy`. The preview is a few 480p
frames a second over localhost, so the encoder's speed is irrelevant next to the
render itself.

Rendering headless is the other half, and it is where the sharp edge is. MuJoCo
picks its GL backend from `MUJOCO_GL`, and it reads that variable at **import**
time, not at render time: `import mujoco` eagerly imports the matching context
module. So an unusable choice does not degrade to a missing preview, it makes
`import mujoco` raise, and the whole simulator backend disappears with it.

EGL is the right default for a server that may be reached over SSH, since it
needs no display. Whether it *works* has two independent ways to fail, and both
were hit on this machine:

  1. MuJoCo's EGL module dereferences `OpenGL.EGL.EGLDeviceEXT`, which PyOpenGL
     3.1.0 does not define -- `import mujoco` dies with `AttributeError: module
     'OpenGL.EGL' has no attribute 'EGLDeviceEXT'`.
  2. Even where that attribute exists, the EGL *driver* may not support the
     `PLATFORM_DEVICE` extension, and the renderer fails to construct.

Neither can be checked in this process. MuJoCo sets `PYOPENGL_PLATFORM=egl`
itself immediately before importing `OpenGL.EGL`; anything that imports that
module first -- including a well-meaning check for `EGLDeviceEXT` -- binds
PyOpenGL to the GLX platform instead (whenever `DISPLAY` is set), after which
MuJoCo's EGL device query fails. The check breaks what it is checking, and
PyOpenGL's platform binding cannot be undone once made.

So `probe_gl_backend()` asks a **subprocess** to actually render a frame under
each candidate, and the winner is the one that did. One process launch at
startup, and the answer covers both failure modes instead of approximating one
of them.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

# In preference order. EGL first because a control-room GUI may be driven over
# SSH; GLFW next because it works on any machine with a display; OSMesa last
# because software rendering is slow but beats no preview at all.
GL_CANDIDATES = ("egl", "glfw", "osmesa")

# Renders a frame from a model with no external assets, so the probe tests the
# GL stack and nothing else.
_PROBE = """
import mujoco
model = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>"
)
renderer = mujoco.Renderer(model, height=16, width=16)
renderer.update_scene(mujoco.MjData(model))
renderer.render()
renderer.close()
"""


def _probe_one(backend: str, timeout: float) -> str:
    """Try to render under `backend` in a fresh process. Returns "" on success."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", _PROBE],
            env={**os.environ, "MUJOCO_GL": backend},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timed out"
    except OSError as exc:
        return f"could not launch a probe: {exc}"
    if done.returncode == 0:
        return ""
    # The last traceback line is the exception; the rest is noise here.
    lines = [line for line in done.stderr.strip().splitlines() if line.strip()]
    return lines[-1] if lines else f"exit {done.returncode}"


def probe_gl_backend(
    candidates: tuple[str, ...] = GL_CANDIDATES, timeout: float = 30.0
) -> tuple[str, str]:
    """Find a GL backend that renders here. Returns `(backend, why)`.

    `("", why)` means none of them did, and `why` lists what each said.
    """
    failures = []
    for backend in candidates:
        error = _probe_one(backend, timeout)
        if not error:
            return backend, "verified by rendering a test frame"
        failures.append(f"{backend}: {error}")
    return "", "; ".join(failures)


def ensure_headless_gl(probe: bool = True) -> tuple[str, str]:
    """Pick MuJoCo's GL backend. Returns `(backend, why)` for logging.

    Never overrides an explicit `MUJOCO_GL`: an operator who exported one has
    made a decision, and a web server is not the place to overrule it -- nor to
    spend a second probing around it. Must be called before the first
    `import mujoco`, since MuJoCo reads the variable at import time.

    `probe=False` skips the subprocess and leaves the choice to MuJoCo, for
    callers that would rather start instantly than know.
    """
    explicit = os.environ.get("MUJOCO_GL")
    if explicit:
        return explicit, "set in the environment"
    if not probe:
        return "auto", "left to MuJoCo's own detection (probe disabled)"

    backend, why = probe_gl_backend()
    if backend:
        os.environ["MUJOCO_GL"] = backend
        return backend, why
    # Leave the variable unset. Setting a backend we just watched fail would
    # only swap a clear "no GL here" for a confusing import error later.
    return "", f"no working GL backend ({why})"


class SimPreview:
    """Renders a MuJoCo model from a *copy* of the live simulation state.

    Two constraints shape this, and they pull in opposite directions.

    **The control loop must not wait for a render.** Pointing a renderer at
    `sim.data` and holding the backend lock while it draws costs tens of
    milliseconds out of the loop's 33 ms budget -- measured at the time as
    control steps dropping to 12 Hz whenever a browser had the preview open. So
    the lock is held only for a `qpos` memcpy, and the draw runs against this
    object's own `MjData`. The preview is then a frame or two stale, which is
    invisible at 10 fps and is the right trade.

    **A GL context belongs to a thread, not to one caller at a time.** This is
    the subtler one, and a mutex does not satisfy it. `ThreadingHTTPServer`
    serves each request on a fresh thread, so without care every render after
    the first runs on a thread the context is not current on. EGL raises
    (`EGL_BAD_ACCESS` from `eglMakeCurrent`); GLFW only warns and lets MuJoCo
    read the framebuffer anyway, which returns tearing, chroma noise and stale
    bands -- intermittently, because whether the previous request's thread has
    exited and released the binding is a race.

    So every GL call -- constructing the renderer, drawing, closing it -- is
    submitted to a single long-lived worker thread that owns the context for the
    life of this object. Callers get a plain blocking method and never see it.
    """

    def __init__(self, model: object) -> None:
        import mujoco

        self._mujoco = mujoco
        self._model = model
        self._data = mujoco.MjData(model)
        self._renderer: object | None = None
        self._size: tuple[int, int] | None = None
        # max_workers=1 is the whole point: one thread, created on first submit
        # and reused for every job after, so the context is made current on the
        # thread that already holds it.
        self._gl = ThreadPoolExecutor(max_workers=1, thread_name_prefix="so-snake-gl")

    def frame(
        self, camera: str, width: int, height: int, capture: Callable[[object], None]
    ) -> np.ndarray:
        """Capture the live state and draw it, both on the GL owner thread.

        `capture` is handed this object's `MjData` and should copy the live
        simulation into it -- taking whatever lock guards the live one. Doing it
        inside the job rather than in the caller keeps the copy ordered against
        the draw without a second lock, and keeps the caller's lock hold to the
        memcpy.
        """
        return self._gl.submit(self._capture_and_render, camera, width, height, capture).result()

    def _capture_and_render(
        self, camera: str, width: int, height: int, capture: Callable[[object], None]
    ) -> np.ndarray:
        capture(self._data)
        if self._renderer is None or self._size != (width, height):
            self._close_renderer()
            self._renderer = self._mujoco.Renderer(self._model, height=height, width=width)
            self._size = (width, height)
        # The copied qpos has not been through forward kinematics in this
        # MjData, so site and geom transforms would otherwise be whatever the
        # last render left behind.
        self._mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=camera)
        # Copied: MuJoCo hands back a view of a buffer it reuses next render.
        return np.array(self._renderer.render())

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._size = None

    def close(self) -> None:
        """Release the renderer and the worker. Safe to call from any thread."""
        try:
            self._gl.submit(self._close_renderer).result(timeout=10.0)
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass
        finally:
            self._gl.shutdown(wait=False)


def encode_png(image: np.ndarray) -> bytes:
    """Encode an `(H, W, 3)` uint8 array as a PNG.

    Filter type 0 (None) on every scanline. Choosing per-line filters would
    compress better, but the frames are going over loopback to a browser that
    asked for them a sixteenth of a second ago; bytes are not the constraint.
    """
    array = np.ascontiguousarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) uint8 image, got shape {array.shape}")
    height, width = array.shape[:2]

    # Each scanline is prefixed with its filter byte, then the whole lot is
    # deflated as one stream.
    raw = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), array.reshape(height, width * 3)], axis=1
    ).tobytes()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB, no interlace
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", header),
            chunk(b"IDAT", zlib.compress(raw, 6)),
            chunk(b"IEND", b""),
        ]
    )


def placeholder_png(width: int, height: int, message: str = "") -> bytes:
    """A flat dark frame, for when there is no simulator to render.

    Returning an image rather than a 404 keeps the browser's `<img>` from
    flashing its broken-image icon every poll while the mock backend is
    selected; the UI says what is going on in text next to it.
    """
    del message  # drawing text without a font library is not worth it
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    image[:, :, 2] = 34
    return encode_png(image)
