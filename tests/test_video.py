"""Video encoding: choosing an encoder, and keeping frames lined up with rows.

The encoder choice is verified against fakes and against this machine; the
recording tests run a real loop with a fake camera at a small resolution, so
they exercise the whole path -- open, submit per step, close, meta -- without
needing a camera or a second of wall clock.
"""

from __future__ import annotations

import numpy as np
import pytest

from so_snake.config import SoSnakeConfig
from so_snake.data import EpisodeRecorder, VideoConfig, probe_encoder, read_meta, select_encoder
from so_snake.data.video import HW_ENCODERS, VideoWriter
from so_snake.rig import RigSpec, build_backend, build_source
from so_snake.teleop.loop import TeleopLoop

# Small enough to encode in milliseconds, large enough for every encoder's
# minimum dimensions.
W, H = 160, 120


@pytest.fixture
def config() -> SoSnakeConfig:
    return SoSnakeConfig()


class FakeRig:
    """A camera rig that hands out frames without a camera.

    `blanks` is the set of step indices at which the camera has nothing fresh,
    which is what a real one does whenever the loop outruns it.
    """

    def __init__(self, roles=("wrist",), blanks=(), width=W, height=H) -> None:
        self.roles = tuple(roles)
        self.blanks = set(blanks)
        self.width, self.height = width, height
        self.reads = 0

    def read_latest(self, role: str):
        if role not in self.roles:
            return None
        index = self.reads
        self.reads += 1
        if index in self.blanks:
            return None
        rng = np.random.default_rng(index)
        return rng.integers(0, 256, (self.height, self.width, 3), dtype=np.uint8)


def record(root, config, *, steps=40, cameras=None, keep=True, video=None):
    spec = RigSpec(backend="mock", source="scripted", scripted_steps=steps + 10)
    backend, source = build_backend(spec, config), build_source(spec, config)
    loop = TeleopLoop(source, backend, config)
    recorder = EpisodeRecorder(
        root, config=config, backend="mock", source="scripted",
        joint_names=backend.joint_names, cameras=cameras, video=video,
    )
    recorder.start(task="test")
    loop.run(max_steps=steps, realtime=False, on_step=recorder.append)
    return recorder.stop(keep=keep)


# ------------------------------------------------------------------- selection


def test_many_cores_keeps_the_smaller_files():
    choice = select_encoder(width=W, height=H, cpu_count=32)
    assert not choice.hardware
    assert "32 CPUs" in choice.reason


def test_few_cores_takes_the_hardware_encoder_if_one_works():
    choice = select_encoder(width=W, height=H, cpu_count=2)
    # Only assert the rule, not the outcome: a machine with no hardware encoder
    # must still return something, and say that is why.
    if choice.hardware:
        assert choice.codec in HW_ENCODERS
        assert "2 CPUs is under" in choice.reason
    else:
        assert "no hardware encoder works here" in choice.reason


def test_an_explicitly_named_encoder_is_still_verified():
    """Naming an encoder that cannot run here is an error, not a silent swap."""
    with pytest.raises(RuntimeError, match="cannot encode here"):
        select_encoder(VideoConfig(codec="definitely_not_a_codec"), width=W, height=H)


def test_the_probe_encodes_rather_than_just_constructing():
    """A codec that exists but cannot open must fail the probe, not pass it."""
    assert probe_encoder("libsvtav1", W, H) == ""
    assert probe_encoder("no_such_encoder", W, H) != ""


def test_the_chosen_encoder_is_reported_with_a_reason():
    choice = select_encoder(width=W, height=H)
    assert choice.codec and choice.reason
    assert choice.to_json()["reason"] == choice.reason


# -------------------------------------------------------------------- recording


def test_recording_with_cameras_writes_a_video_per_role(tmp_path, config):
    meta = record(tmp_path, config, steps=30, cameras=FakeRig(roles=("wrist", "third_person")))
    directory = tmp_path / meta.id
    assert (directory / "wrist.mp4").is_file()
    assert (directory / "third_person.mp4").is_file()
    assert (directory / "wrist.mp4").stat().st_size > 0


def test_meta_records_the_codec_and_why(tmp_path, config):
    meta = record(tmp_path, config, steps=20, cameras=FakeRig())
    encoder = read_meta(tmp_path / meta.id).video["encoder"]
    assert encoder["codec"]
    assert encoder["reason"]
    assert isinstance(encoder["hardware"], bool)


def test_one_video_frame_per_control_step(tmp_path, config):
    """Video frame i is row i. Losing that is invisible until training."""
    meta = record(tmp_path, config, steps=30, cameras=FakeRig())
    counts = read_meta(tmp_path / meta.id).video["cameras"]["wrist"]
    assert counts["written"] + counts["dropped"] == meta.n_steps


def test_a_camera_with_nothing_fresh_repeats_rather_than_skips(tmp_path, config):
    """Skipping would shift every later frame against its row."""
    meta = record(tmp_path, config, steps=30, cameras=FakeRig(blanks=range(5, 15)))
    counts = read_meta(tmp_path / meta.id).video["cameras"]["wrist"]
    assert counts["stale"] > 0
    assert counts["written"] + counts["dropped"] == meta.n_steps


def test_the_video_decodes_to_the_frames_that_were_recorded(tmp_path, config):
    av = pytest.importorskip("av")
    meta = record(tmp_path, config, steps=25, cameras=FakeRig())
    with av.open(str(tmp_path / meta.id / "wrist.mp4")) as container:
        decoded = [f for f in container.decode(container.streams.video[0])]
    assert len(decoded) == meta.n_steps
    assert (decoded[0].height, decoded[0].width) == (H, W)


def test_a_discarded_take_leaves_no_footage(tmp_path, config):
    """Video the operator threw away must not survive without a meta to explain it."""
    assert record(tmp_path, config, steps=20, cameras=FakeRig(), keep=False) is None
    assert list(tmp_path.iterdir()) == []


def test_recording_without_cameras_still_works(tmp_path, config):
    meta = record(tmp_path, config, steps=20, cameras=None)
    assert meta.video == {}
    assert not list((tmp_path / meta.id).glob("*.mp4"))


def test_a_camera_that_has_produced_nothing_is_left_out(tmp_path, config):
    """Its frame size is unknown, and waiting would block the record button."""
    rig = FakeRig(roles=("wrist",), blanks=range(0, 10_000))
    meta = record(tmp_path, config, steps=20, cameras=rig)
    assert meta.video == {}


# ---------------------------------------------------------------------- writer


def test_the_queue_is_bounded_so_a_slow_encoder_cannot_eat_memory(tmp_path, config):
    """1080p RGB is 6 MB a frame; an unbounded backlog is the failure mode."""
    choice = select_encoder(width=W, height=H)
    writer = VideoWriter(tmp_path / "w.mp4", choice, W, H, 30.0, VideoConfig(queue_size=4))
    rng = np.random.default_rng(0)
    for _ in range(200):
        writer.submit(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    stats = writer.close()
    assert stats.written + stats.dropped == 200
    assert writer.error == ""
