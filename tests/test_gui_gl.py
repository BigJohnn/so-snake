"""GL backend selection, which decides whether MuJoCo imports at all.

MuJoCo reads `MUJOCO_GL` when it is imported, not when it renders, and eagerly
imports the matching context module -- so an unusable choice does not cost a
camera preview, it makes `import mujoco` raise and takes the simulator with it.

The rule these tests pin is that the choice is made by *rendering*, in a
subprocess, rather than by inspecting anything in this process. That is not
fastidiousness. MuJoCo sets `PYOPENGL_PLATFORM=egl` immediately before importing
`OpenGL.EGL`; a check that imports that module first binds PyOpenGL to GLX
(whenever `DISPLAY` is set) and MuJoCo's EGL device query then fails. An
in-process probe breaks the thing it is probing, and the binding is permanent
once made.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

from so_snake.gui import preview


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)


def test_an_explicit_choice_is_never_probed_around(monkeypatch):
    """An operator who exported MUJOCO_GL has already decided.

    And must not pay a second of subprocess launches to be told so.
    """
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.setattr(
        preview, "probe_gl_backend", lambda *a, **k: pytest.fail("probed despite explicit choice")
    )

    backend, why = preview.ensure_headless_gl()
    assert backend == "glfw"
    assert "environment" in why
    assert os.environ["MUJOCO_GL"] == "glfw"


def test_the_first_backend_that_renders_wins(monkeypatch):
    monkeypatch.setattr(preview, "probe_gl_backend", lambda *a, **k: ("glfw", "verified"))

    backend, _ = preview.ensure_headless_gl()
    assert backend == "glfw"
    assert os.environ["MUJOCO_GL"] == "glfw"


def test_the_variable_is_left_unset_when_nothing_renders(monkeypatch):
    """Setting a backend we just watched fail buys a confusing import error.

    Leaving it unset keeps the failure where it is legible: no preview, and a
    startup line saying no GL backend works here.
    """
    monkeypatch.setattr(preview, "probe_gl_backend", lambda *a, **k: ("", "egl: boom; glfw: boom"))

    backend, why = preview.ensure_headless_gl()
    assert backend == ""
    assert "MUJOCO_GL" not in os.environ
    assert "no working GL backend" in why
    assert "egl: boom" in why


def test_probing_can_be_skipped(monkeypatch):
    monkeypatch.setattr(
        preview, "probe_gl_backend", lambda *a, **k: pytest.fail("probed despite probe=False")
    )
    backend, why = preview.ensure_headless_gl(probe=False)
    assert backend == "auto"
    assert "MUJOCO_GL" not in os.environ
    assert "disabled" in why


def test_probe_tries_every_candidate_in_order_and_stops_at_the_first_success(monkeypatch):
    tried: list[str] = []

    def fake(backend: str, timeout: float) -> str:
        tried.append(backend)
        return "" if backend == "glfw" else "no such device"

    monkeypatch.setattr(preview, "_probe_one", fake)
    backend, why = preview.probe_gl_backend(("egl", "glfw", "osmesa"))

    assert backend == "glfw"
    assert tried == ["egl", "glfw"]  # osmesa never reached
    assert "rendering" in why


def test_probe_reports_what_each_candidate_said_when_all_fail(monkeypatch):
    monkeypatch.setattr(preview, "_probe_one", lambda backend, timeout: f"{backend} is unhappy")
    backend, why = preview.probe_gl_backend(("egl", "glfw"))

    assert backend == ""
    assert "egl: egl is unhappy" in why
    assert "glfw: glfw is unhappy" in why


def test_probe_runs_in_a_subprocess_with_the_backend_in_its_environment(monkeypatch):
    """The isolation is the whole point, so it is asserted rather than assumed."""
    seen: dict = {}

    class Done:
        returncode = 0
        stderr = ""

    def fake_run(argv, env, **kwargs):
        seen["argv"], seen["env"] = argv, env
        return Done()

    monkeypatch.setattr(preview.subprocess, "run", fake_run)
    assert preview._probe_one("egl", timeout=5.0) == ""

    assert seen["argv"][0] == preview.sys.executable
    assert seen["env"]["MUJOCO_GL"] == "egl"
    assert "mujoco.Renderer" in seen["argv"][2]


def test_probe_survives_a_hung_backend(monkeypatch):
    """A wedged GL driver must not hang the server's startup forever."""

    def fake_run(*args, **kwargs):
        raise preview.subprocess.TimeoutExpired(cmd="probe", timeout=1.0)

    monkeypatch.setattr(preview.subprocess, "run", fake_run)
    assert preview._probe_one("egl", timeout=1.0) == "timed out"


def _sim_preview():
    """A `SimPreview` over the real model, or a skip if this box cannot render."""
    pytest.importorskip("mujoco")
    backend, why = preview.probe_gl_backend()
    if not backend:
        pytest.skip(f"no working GL backend here ({why})")
    os.environ.setdefault("MUJOCO_GL", backend)

    from so_snake.rig import RigSpec, build_backend

    arm = build_backend(RigSpec(backend="mujoco"))
    arm.connect()

    def capture(dest):
        dest.qpos[:] = arm.sim.data.qpos
        dest.qvel[:] = arm.sim.data.qvel

    return preview.SimPreview(arm.sim.model), capture


def test_every_gl_call_lands_on_one_thread():
    """The invariant a mutex does not give you.

    A GL context belongs to a thread, not to one caller at a time, and
    `ThreadingHTTPServer` serves each request on a fresh thread. Rendering from
    whichever thread happened to ask returns tearing and stale bands under GLFW
    and raises `EGL_BAD_ACCESS` under EGL.
    """
    sim_preview, capture = _sim_preview()
    threads: set[int] = set()

    def recording_capture(dest):
        threads.add(threading.get_ident())
        capture(dest)

    try:
        sim_preview.frame("wrist", 64, 48, recording_capture)
        for _ in range(6):
            worker = threading.Thread(
                target=lambda: sim_preview.frame("wrist", 64, 48, recording_capture)
            )
            worker.start()
            worker.join()
    finally:
        sim_preview.close()

    assert len(threads) == 1, f"GL work ran on {len(threads)} threads: {threads}"
    assert threading.get_ident() not in threads, "GL work must not run on the caller's thread"


def test_frames_from_fresh_threads_are_not_corrupted():
    """The user-visible symptom, pinned.

    Renders an unchanged scene from a series of fresh threads -- what the HTTP
    server does -- and requires every frame to match one taken on the owner
    thread. Before the single-threaded executor these came back near-black
    (mean 0.00 against a reference of 91.23); the tolerance below is far tighter
    than that and far looser than the renderer's own +/-1 dithering on a
    handful of pixels.
    """
    sim_preview, capture = _sim_preview()
    try:
        reference = sim_preview.frame("wrist", 160, 120, capture).astype(int)
        frames = []
        for _ in range(6):
            worker = threading.Thread(
                target=lambda: frames.append(sim_preview.frame("wrist", 160, 120, capture))
            )
            worker.start()
            worker.join()
    finally:
        sim_preview.close()

    for i, frame in enumerate(frames):
        drift = float(np.abs(frame.astype(int) - reference).mean())
        assert drift < 1.0, f"frame {i} from a fresh thread drifted by {drift:.2f} per pixel"


def test_a_nonsense_backend_is_reported_as_failing():
    """The probe against the real MuJoCo, unmocked.

    Everything above stubs the subprocess, which would happily pass if the probe
    script itself were broken. This one runs it, and asserts the negative case
    -- no GL stack answers to "definitely-not-a-gl-backend" -- so it needs no
    display and holds on a headless CI box.
    """
    pytest.importorskip("mujoco")
    error = preview._probe_one("definitely-not-a-gl-backend", timeout=60.0)
    assert error, "an invalid backend must be reported as a failure"
