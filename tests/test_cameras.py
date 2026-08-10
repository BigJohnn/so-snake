"""The camera rig: role assignment, lifecycle, and what a stalled camera does.

No hardware here. `CameraRig` takes a factory for opening one camera, so the
lifecycle -- including the rollback when the second of two cameras fails -- is
exercised against fakes. The parts that genuinely need a device (enumeration,
lerobot's capture thread) are verified on the bench, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from so_snake.gui.server import cameras_from_body, spec_from_body
from so_snake.m0_perception import CameraRig, CameraSpec, frame_for_preview
from so_snake.rig import RigSpec, build_cameras


class FakeCamera:
    """Stands in for a connected `OpenCVCamera`."""

    def __init__(self, frame: np.ndarray | None = None, stale: bool = False) -> None:
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8) if frame is None else frame
        self.stale = stale
        self.disconnected = False

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        if self.stale:
            # What lerobot raises when the capture thread has stopped feeding it.
            raise TimeoutError("latest frame is too old")
        return self.frame

    def disconnect(self) -> None:
        self.disconnected = True


def opener(**by_role: FakeCamera):
    """A factory that hands out the camera registered for each role."""

    def open_camera(spec: CameraSpec) -> FakeCamera:
        camera = by_role.get(spec.role)
        if camera is None:
            raise RuntimeError(f"no fake camera for {spec.role}")
        return camera

    return open_camera


# ----------------------------------------------------------------- specs


def test_role_must_be_one_the_system_knows():
    with pytest.raises(ValueError, match="camera role"):
        CameraSpec(role="over_the_shoulder", index_or_path=0).validate()


def test_two_cameras_cannot_share_a_role():
    """The second assignment would silently win, and the pane would lie."""
    with pytest.raises(ValueError, match="same role"):
        CameraRig((CameraSpec("wrist", 0), CameraSpec("wrist", 1)))


def test_rig_spec_carries_cameras_through_validation():
    spec = RigSpec(backend="mock", cameras=(CameraSpec("wrist", 3),))
    spec.validate()
    assert build_cameras(spec).roles == ("wrist",)


def test_a_rig_with_no_cameras_still_works():
    """Callers should not have to branch on None to find out they have none."""
    rig = build_cameras(RigSpec(backend="mock"))
    rig.connect()
    assert rig.read_latest("wrist") is None
    rig.disconnect()


# ------------------------------------------------------------- lifecycle


def test_connect_opens_every_assigned_role():
    wrist, third = FakeCamera(), FakeCamera()
    rig = CameraRig(
        (CameraSpec("wrist", 0), CameraSpec("third_person", 1)),
        open_camera=opener(wrist=wrist, third_person=third),
    )
    rig.connect()
    assert rig.is_connected
    assert sorted(rig.status()["connected"]) == ["third_person", "wrist"]
    rig.disconnect()
    assert wrist.disconnected and third.disconnected
    assert not rig.is_connected


def test_a_half_opened_rig_closes_what_it_opened():
    """A leaked handle is found later, by the next session failing to open it."""
    wrist = FakeCamera()

    def open_camera(spec: CameraSpec) -> FakeCamera:
        if spec.role == "wrist":
            return wrist
        raise RuntimeError("device busy")

    rig = CameraRig(
        (CameraSpec("wrist", 0), CameraSpec("third_person", 1)), open_camera=open_camera
    )
    with pytest.raises(RuntimeError, match="device busy"):
        rig.connect()
    assert wrist.disconnected
    assert not rig.is_connected


def test_disconnect_survives_a_camera_that_throws_on_close():
    class Rude(FakeCamera):
        def disconnect(self) -> None:
            raise OSError("device went away")

    third = FakeCamera()
    rig = CameraRig(
        (CameraSpec("wrist", 0), CameraSpec("third_person", 1)),
        open_camera=opener(wrist=Rude(), third_person=third),
    )
    rig.connect()
    rig.disconnect()  # must not raise; teardown has to keep going
    assert third.disconnected


# ----------------------------------------------------------------- reads


def test_read_latest_returns_the_frame():
    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    rig = CameraRig((CameraSpec("wrist", 0),), open_camera=opener(wrist=FakeCamera(frame)))
    rig.connect()
    assert np.array_equal(rig.read_latest("wrist"), frame)


def test_every_not_right_now_reads_as_no_frame():
    """A stalled camera, an unassigned role and a closed rig are all None.

    None of them is a reason to raise into a caller that may be the control
    loop: the answer to "what does the camera see" is "nothing usable", and the
    step still has to happen on time.
    """
    rig = CameraRig((CameraSpec("wrist", 0),), open_camera=opener(wrist=FakeCamera(stale=True)))
    rig.connect()
    assert rig.read_latest("wrist") is None  # stalled
    assert rig.read_latest("third_person") is None  # never assigned
    rig.disconnect()
    assert rig.read_latest("wrist") is None  # closed


# ------------------------------------------------------------- preview fit


def test_preview_letterboxes_rather_than_stretching():
    """Framing is what the operator judges from the pane; squashing misleads."""
    tall = np.full((100, 50, 3), 255, dtype=np.uint8)
    out = frame_for_preview(tall, 640, 480)
    assert out.shape == (480, 640, 3)
    # 2:1 source in a 4:3 pane: full height, bars left and right.
    assert out[:, 0].max() == 0 and out[:, -1].max() == 0
    assert out[240, 320].max() == 255


def test_preview_handles_an_empty_frame_without_dividing_by_zero():
    assert frame_for_preview(np.zeros((0, 0, 3), np.uint8), 32, 24).shape == (24, 32, 3)


# ------------------------------------------------------------ request body


def test_numeric_strings_are_indices_and_blanks_are_omitted():
    """The picker posts strings; an unset role posts "" and means "not this session"."""
    specs = cameras_from_body({"cameras": {"wrist": "3", "third_person": ""}})
    assert specs == (CameraSpec("wrist", 3),)


def test_paths_survive_as_paths():
    specs = cameras_from_body({"cameras": {"wrist": "/dev/video0"}})
    assert specs == (CameraSpec("wrist", "/dev/video0"),)


@pytest.mark.parametrize(
    "body",
    [
        {"cameras": {"wrist": True}},  # bool is an int subclass, but not a device
        {"cameras": [0, 1]},
        {"cameras": {"elbow": 0}},
    ],
)
def test_a_bad_camera_body_is_rejected(body):
    with pytest.raises(ValueError):
        spec_from_body({"backend": "mock", **body})


def test_devices_are_listed_without_invented_names(monkeypatch):
    """macOS has no dependable index -> name mapping, so none is claimed.

    An earlier version zipped `system_profiler`'s ordering onto the OpenCV
    index and reported the built-in webcam as a USB camera. Both are 1080p and
    both deliver frames, so nothing downstream could notice.
    """
    import so_snake.m0_perception.cameras as mod

    class FakeCapture:
        def __init__(self, index):
            self.index = index

        def isOpened(self):  # noqa: N802 - cv2 API
            return self.index < 2

        def read(self):
            return True, np.full((8, 8, 3), 10 * (self.index + 1), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(mod.platform, "system", lambda: "Darwin")
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    devices = mod.list_devices(max_index=4)

    assert [d["index"] for d in devices] == [0, 1]
    assert all(d["name"] == "" for d in devices)
    assert all(d["thumbnail"].startswith("data:image/jpeg;base64,") for d in devices)
