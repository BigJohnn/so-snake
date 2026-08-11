"""Finding the arm's port and the cameras, without either being attached.

No hardware here, and none of these tests opens a device: enumeration is faked
at the `list_serial_ports` / `scan_cameras` seam, and the servo ping at
`probe_port`. What is being pinned down is the judgement -- which ports are
never the arm, when detection is allowed to answer, and when it must refuse
rather than guess -- because that judgement is what runs unattended on a bench
where picking the wrong device drives the arm.
"""

from __future__ import annotations

import pytest

import so_snake.devices as devices
from so_snake.devices import (
    ARM_PORT_ENV,
    CameraDevice,
    DeviceDetectionError,
    SerialPortInfo,
    arm_port_candidates,
    detect_arm_port,
    resolve_camera_specs,
)

# The board on this bench: a WCH CH343 bridge, reported by pyserial as below.
ARM = SerialPortInfo(
    device="/dev/cu.usbmodem58760434321",
    description="USB Single Serial",
    vid=0x1A86,
    pid=0x55D3,
    serial_number="5876043432",
)
BLUETOOTH = SerialPortInfo(device="/dev/cu.Bluetooth-Incoming-Port")
DEBUG = SerialPortInfo(device="/dev/cu.debug-console")
OTHER_USB = SerialPortInfo(device="/dev/cu.usbserial-1420", description="FT232R", vid=0x0403, pid=0x6001)
LINUX_ARM = SerialPortInfo(device="/dev/ttyACM0")


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch):
    """The bench sets SO_SNAKE_ARM_PORT sometimes; these tests must not see it."""
    monkeypatch.delenv(ARM_PORT_ENV, raising=False)


def fake_ports(monkeypatch, *ports: SerialPortInfo) -> None:
    monkeypatch.setattr(devices, "list_serial_ports", lambda: list(ports))


# ---------------------------------------------------------------- candidates


def test_bluetooth_and_debug_ports_are_never_the_arm():
    """Both are present on every macOS machine, and opening Bluetooth blocks."""
    assert BLUETOOTH.score < 0
    assert DEBUG.score < 0
    assert arm_port_candidates([ARM, BLUETOOTH, DEBUG]) == [ARM]


def test_a_known_bridge_chip_outranks_a_name_that_merely_looks_right():
    """USB ids are what the board is; a device name is what the OS called it."""
    assert ARM.score > LINUX_ARM.score
    assert arm_port_candidates([LINUX_ARM, ARM])[0] is ARM
    assert ARM.known_as.startswith("WCH")
    assert ARM.usb_id == "1a86:55d3"


def test_a_port_with_no_usb_layer_is_still_a_candidate_on_linux():
    """pyserial reports no vid/pid for many /dev/ttyACM devices; those still count."""
    assert LINUX_ARM.score > 0
    assert arm_port_candidates([LINUX_ARM, BLUETOOTH]) == [LINUX_ARM]


# ----------------------------------------------------------------- detection


def test_an_explicit_port_is_never_second_guessed(monkeypatch):
    """Detection must not override the operator, even when it disagrees."""
    fake_ports(monkeypatch, ARM)
    assert detect_arm_port("/dev/ttyUSB7") == "/dev/ttyUSB7"


def test_the_environment_override_beats_detection(monkeypatch):
    fake_ports(monkeypatch, ARM)
    monkeypatch.setenv(ARM_PORT_ENV, "/dev/ttyUSB9")
    assert detect_arm_port() == "/dev/ttyUSB9"
    # ...but an explicit flag still beats the environment.
    assert detect_arm_port("/dev/ttyUSB1") == "/dev/ttyUSB1"


def test_one_candidate_resolves_without_touching_the_bus(monkeypatch):
    """The common case must not open anything: a probe could disturb a session."""
    fake_ports(monkeypatch, DEBUG, ARM, BLUETOOTH)

    def forbidden(*args, **kwargs):
        raise AssertionError("probe_port must not be called when there is one candidate")

    monkeypatch.setattr(devices, "probe_port", forbidden)
    assert detect_arm_port() == ARM.device
    assert detect_arm_port("auto") == ARM.device
    assert detect_arm_port("") == ARM.device


def test_two_candidates_are_separated_by_which_one_has_servos(monkeypatch):
    fake_ports(monkeypatch, ARM, OTHER_USB)
    monkeypatch.setattr(
        devices, "probe_port", lambda device, **kw: [1, 2, 3] if device == OTHER_USB.device else []
    )
    assert detect_arm_port() == OTHER_USB.device


def test_two_candidates_and_no_answer_refuses_and_lists_them(monkeypatch):
    """Picking the higher-scoring one would be a coin flip that drives an arm."""
    fake_ports(monkeypatch, ARM, OTHER_USB)
    monkeypatch.setattr(devices, "probe_port", lambda device, **kw: [])
    with pytest.raises(DeviceDetectionError) as exc:
        detect_arm_port()
    assert ARM.device in str(exc.value) and OTHER_USB.device in str(exc.value)


def test_two_answering_ports_refuse_rather_than_pick_one(monkeypatch):
    fake_ports(monkeypatch, ARM, OTHER_USB)
    monkeypatch.setattr(devices, "probe_port", lambda device, **kw: [1])
    with pytest.raises(DeviceDetectionError):
        detect_arm_port()


def test_no_candidate_says_so_and_shows_what_was_there(monkeypatch):
    """The listing is the point: "no arm" and "arm not powered" look the same."""
    fake_ports(monkeypatch, BLUETOOTH, DEBUG)
    with pytest.raises(DeviceDetectionError) as exc:
        detect_arm_port()
    assert BLUETOOTH.device in str(exc.value)


def test_detection_never_returns_an_empty_port(monkeypatch):
    """An empty port reaches lerobot as a confusing open failure, so it raises."""
    fake_ports(monkeypatch)
    with pytest.raises(DeviceDetectionError):
        detect_arm_port("")


# ------------------------------------------------------------------ cameras


CAM0 = CameraDevice(index=0, device=0, width=1920, height=1080)
CAM3 = CameraDevice(index=3, device=3, width=640, height=480)


def test_explicit_camera_devices_do_not_trigger_a_scan(monkeypatch):
    """A scan opens every camera on the machine; a fully specified rig pays nothing."""
    monkeypatch.setattr(
        devices, "scan_cameras", lambda *a, **k: (_ for _ in ()).throw(AssertionError("scanned"))
    )
    specs = resolve_camera_specs({"wrist": 3, "third_person": "/dev/video0"})
    assert {s.role: s.index_or_path for s in specs} == {"wrist": 3, "third_person": "/dev/video0"}


def test_a_numeric_string_is_an_index(monkeypatch):
    """It arrives as text from a flag or a form field; OpenCV needs the int."""
    specs = resolve_camera_specs({"wrist": "3"}, devices=[])
    assert specs[0].index_or_path == 3


def test_auto_fills_the_role_when_there_is_exactly_one_camera():
    specs = resolve_camera_specs({"wrist": "auto"}, devices=[CAM0])
    assert (specs[0].role, specs[0].index_or_path) == ("wrist", 0)


def test_auto_refuses_when_more_than_one_camera_could_be_meant():
    """A mislabelled view is silent -- it is found by whoever watches the episode."""
    with pytest.raises(DeviceDetectionError) as exc:
        resolve_camera_specs({"wrist": "auto"}, devices=[CAM0, CAM3])
    assert "index 0" in str(exc.value) and "index 3" in str(exc.value)


def test_auto_ignores_a_camera_another_role_already_claimed():
    """Two roles cannot share a device, so the claimed one is not a candidate."""
    specs = resolve_camera_specs({"third_person": 0, "wrist": "auto"}, devices=[CAM0, CAM3])
    assert {s.role: s.index_or_path for s in specs} == {"third_person": 0, "wrist": 3}


def test_auto_with_no_cameras_at_all_is_an_error_not_an_empty_rig():
    """Silently recording no video is the failure this is meant to prevent."""
    with pytest.raises(DeviceDetectionError):
        resolve_camera_specs({"wrist": "auto"}, devices=[])


def test_an_unknown_role_is_rejected_before_anything_is_opened():
    with pytest.raises(ValueError):
        resolve_camera_specs({"overhead": 0}, devices=[CAM0])
