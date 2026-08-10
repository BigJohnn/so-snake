"""Encoding camera frames into an episode's video files.

Two decisions live here, and both were made from measurements on the bench
rather than from what the encoders claim.

## Which encoder

Not a fixed default, because there is no winner. Measured on this machine, two
1080p streams at 30 Hz:

    encoder                CPU to keep up    bitrate    PSNR
    hevc_videotoolbox           0.30 core   1.67 Mbps   43.0 dB
    libsvtav1 (preset 12)       1.44 core   0.52 Mbps   42.8 dB

Hardware costs 4.8x less CPU; software costs 3.2x less disk, at the same
picture. Which one is right depends on what the recording machine is short of,
so `select_encoder` asks: a machine with few cores cannot spare 1.4 of them
next to a 30 Hz control loop, and takes the hardware encoder; a machine with
cores to spare keeps the smaller files. `VideoConfig.codec` overrides the whole
question when the operator has already decided.

Whatever is chosen, the codec **and the reason** go into `meta.json`. An
episode that looks worse than its neighbours should not require guessing which
encoder produced it.

## Why the probe encodes instead of asking

`av.codec.Codec(name, "w")` only says the encoder was compiled in. It is not
the same question as "does it work here": NVENC is in many FFmpeg builds on
machines with no NVIDIA driver, and a VideoToolbox session can be refused at
open time. Constructing succeeds, and the failure surfaces on the first frame
of a take -- which is the one moment this repository is least willing to fail
in. So `probe_encoder` encodes real frames at the real resolution and believes
only what came out. This is the same reasoning as `gui.preview.probe_gl_backend`,
one step weaker: a bad GL choice makes `import mujoco` raise and so has to be
probed in a subprocess, while a bad encoder raises a catchable exception, so
this one can stay in-process.

## The q:v trap

`-q:v` is not an encoder option. It is an FFmpeg *command line* shorthand that
ffmpeg.c turns into `AV_CODEC_FLAG_QSCALE` plus `global_quality = N *
FF_QP2LAMBDA`; VideoToolbox reads the flag. Passing `"q:v"` through PyAV's
options dict matches no AVOption and is dropped in silence -- quality 40 and
quality 25 produce byte-identical files, both at the encoder's default ~9 Mbps.
(lerobot's `datasets/video_utils.py` has this bug today.) The flag and
`global_quality` are set explicitly below, which is what makes the hardware
encoder usable at all.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import VideoConfig

# In preference order. VideoToolbox first because this is a macOS bench; the
# NVIDIA and Intel entries cost nothing to try and are skipped in milliseconds
# where they are absent.
#
# Jetson is deliberately not here: Orin exposes its encoder through V4L2 M2M
# (`nvv4l2h265enc`), not through the Video Codec SDK that `h264_nvenc` needs, so
# listing nvenc there would find nothing and quietly fall back to the CPU. That
# needs a GStreamer path, and is not in scope.
HW_ENCODERS: tuple[str, ...] = (
    "hevc_videotoolbox",
    "h264_videotoolbox",
    "hevc_nvenc",
    "h264_nvenc",
    "h264_vaapi",
    "h264_qsv",
)

# libsvtav1 first: measured 2.8x smaller than libx264 at the same picture, and
# it keeps up (141 fps against the 60 fps two cameras need) where libx264 does
# not (53 fps).
SW_ENCODERS: tuple[str, ...] = ("libsvtav1", "libx264")

# FFmpeg's fixed point scale for `global_quality`. `-q:v N` on the command line
# means `global_quality = N * FF_QP2LAMBDA`.
FF_QP2LAMBDA = 118


@dataclass(frozen=True)
class EncoderChoice:
    """A codec that has been verified to encode here, and why it was picked."""

    codec: str
    reason: str
    hardware: bool

    def to_json(self) -> dict[str, Any]:
        return {"codec": self.codec, "reason": self.reason, "hardware": self.hardware}


@dataclass
class VideoStats:
    """What actually reached the file. All three matter for a training set."""

    written: int = 0
    # Queue was full: the encoder could not keep up and these frames are gone.
    # The video is then shorter than the episode and the row alignment is
    # broken, which is why this is recorded rather than tolerated in silence.
    dropped: int = 0
    # The camera had nothing new, so the previous frame was written again to
    # keep video frame i lined up with row i. The picture is real, just old.
    stale: int = 0

    def to_json(self) -> dict[str, int]:
        return {"written": self.written, "dropped": self.dropped, "stale": self.stale}


def video_import_error() -> str:
    """Why video cannot be encoded here, or "" if it can."""
    try:
        import av  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - the reason is the payload
        return f"{type(exc).__name__}: {exc}"
    return ""


def _apply_options(stream: Any, codec: str, config: VideoConfig) -> None:
    """Set the quality knobs that actually reach each encoder."""
    import av.codec.context as codec_context

    if codec in HW_ENCODERS:
        if config.hw_bitrate > 0:
            stream.codec_context.bit_rate = int(config.hw_bitrate)
            return
        # See the module docstring: the flag is what VideoToolbox reads, and
        # `global_quality` is a generic AVOption so it goes through the dict.
        stream.codec_context.flags |= codec_context.Flags.qscale
        stream.options = {"global_quality": str(int(config.hw_quality * FF_QP2LAMBDA))}
        return

    options = {"crf": str(int(config.crf))}
    if codec == "libsvtav1":
        options["preset"] = str(int(config.preset))
    stream.options = options


def _open_stream(container: Any, codec: str, width: int, height: int,
                 fps: float, config: VideoConfig) -> Any:
    import av

    stream = container.add_stream(codec, rate=int(round(fps)))
    stream.width, stream.height = int(width), int(height)
    stream.pix_fmt = "yuv420p"
    if codec == "hevc_videotoolbox":
        # QuickTime, Preview and Safari want `hvc1`; VideoToolbox tags `hev1` by
        # default, which they handle badly. torchcodec does not care, but a
        # human double-clicking an episode to check a take does.
        stream.codec_tag = "hvc1"
    _apply_options(stream, codec, config)
    del av
    return stream


def probe_encoder(codec: str, width: int = 640, height: int = 480,
                  config: VideoConfig | None = None, frames: int = 3) -> str:
    """Actually encode `frames` frames with `codec`. Returns "" on success.

    Noise rather than a flat colour: a constant image can encode successfully
    through a path that falls over on real content, and it makes a zero-byte
    result look like a win.
    """
    error = video_import_error()
    if error:
        return error

    import io

    import av

    config = config or VideoConfig()
    rng = np.random.default_rng(0)
    try:
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        try:
            stream = _open_stream(container, codec, width, height, 30.0, config)
            for _ in range(frames):
                image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        if buffer.getbuffer().nbytes == 0:
            return "encoder produced no output"
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable here"
        return f"{type(exc).__name__}: {exc}"
    return ""


def select_encoder(
    config: VideoConfig | None = None,
    *,
    width: int = 640,
    height: int = 480,
    cpu_count: int | None = None,
) -> EncoderChoice:
    """Pick an encoder for this machine, verifying it before returning it.

    Raises if nothing works: recording camera frames with no encoder is not
    something to paper over, and the caller can still record an episode without
    video by not asking for any.
    """
    config = config or VideoConfig()
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)

    if config.codec != "auto":
        error = probe_encoder(config.codec, width, height, config)
        if error:
            raise RuntimeError(f"codec {config.codec!r} was requested but cannot encode here: {error}")
        return EncoderChoice(
            codec=config.codec,
            reason="explicitly configured",
            hardware=config.codec in HW_ENCODERS,
        )

    prefer_hardware = cores < config.hw_core_threshold
    tried: list[str] = []

    if prefer_hardware:
        for codec in HW_ENCODERS:
            if not probe_encoder(codec, width, height, config):
                return EncoderChoice(
                    codec=codec,
                    reason=(f"{cores} CPUs is under the {config.hw_core_threshold} needed to "
                            f"spare ~1.4 cores for software encoding, and {codec} verified here"),
                    hardware=True,
                )
            tried.append(codec)

    for codec in SW_ENCODERS:
        if not probe_encoder(codec, width, height, config):
            reason = (
                f"{cores} CPUs is enough to spare ~1.4 cores for software encoding, which "
                f"produces ~3x smaller files at the same picture"
                if not prefer_hardware
                else f"no hardware encoder works here (tried {', '.join(tried)})"
            )
            return EncoderChoice(codec=codec, reason=reason, hardware=False)
        tried.append(codec)

    raise RuntimeError(f"no usable video encoder (tried {', '.join(tried)})")


class VideoWriter:
    """One camera's mp4, encoded on its own thread.

    The control loop calls `submit` and must never wait: encoding a 1080p frame
    is milliseconds and the loop's whole budget is 33. So frames go onto a
    bounded queue and an encoder thread drains it. Bounded, because an unbounded
    one turns an encoder that has fallen behind into memory exhaustion during a
    take -- 1080p RGB is 6 MB a frame, and a minute of backlog is 11 GB.

    Frames are written one per control step even when the camera has produced
    nothing new, so video frame *i* is row *i* of `frames.npz`. Losing that
    alignment would make the episode useless for training in a way that is
    invisible until someone trains on it.
    """

    def __init__(
        self,
        path: Path,
        choice: EncoderChoice,
        width: int,
        height: int,
        fps: float,
        config: VideoConfig | None = None,
    ) -> None:
        import av

        self.path = Path(path)
        self.choice = choice
        self.width, self.height = int(width), int(height)
        self.config = config or VideoConfig()

        self._stats = VideoStats()
        self._stats_lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=self.config.queue_size)
        self._error = ""
        self._last: np.ndarray | None = None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        self._stream = _open_stream(
            self._container, choice.codec, self.width, self.height, fps, self.config
        )
        self._thread = threading.Thread(
            target=self._drain, name=f"so-snake-encode-{self.path.stem}", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ public

    def submit(self, frame: np.ndarray | None) -> None:
        """Hand over this step's frame. Never blocks, never raises.

        `None` means the camera had nothing fresh; the previous frame is written
        again to hold the alignment, and counted as stale. Before any frame has
        arrived there is nothing to repeat, so the step is counted as dropped --
        the video will be short by that much and `meta.json` will say so.
        """
        stale = frame is None
        if stale:
            frame = self._last
        if frame is None:
            with self._stats_lock:
                self._stats.dropped += 1
            return

        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            with self._stats_lock:
                self._stats.dropped += 1
            return

        self._last = frame
        if stale:
            with self._stats_lock:
                self._stats.stale += 1

    def close(self) -> VideoStats:
        """Flush the queue, close the file, and report what reached it."""
        self._queue.put(None)
        self._thread.join(timeout=60.0)
        with self._stats_lock:
            return VideoStats(self._stats.written, self._stats.dropped, self._stats.stale)

    @property
    def error(self) -> str:
        return self._error

    def stats(self) -> VideoStats:
        with self._stats_lock:
            return VideoStats(self._stats.written, self._stats.dropped, self._stats.stale)

    # ----------------------------------------------------------------- private

    def _drain(self) -> None:
        import av

        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                self._encode(av, frame)
            for packet in self._stream.encode():
                self._container.mux(packet)
        except Exception as exc:  # noqa: BLE001 - must not kill the recording
            self._error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self._container.close()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass

    def _encode(self, av: Any, image: np.ndarray) -> None:
        if image.shape[0] != self.height or image.shape[1] != self.width:
            # A camera that changes mode mid-take. Rescaling keeps the file
            # playable and the alignment intact, which beats losing the rest.
            import cv2

            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        for packet in self._stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            self._container.mux(packet)
        with self._stats_lock:
            self._stats.written += 1


@dataclass
class VideoSet:
    """The video files for one episode, one per camera role."""

    choice: EncoderChoice
    writers: dict[str, VideoWriter] = field(default_factory=dict)

    def submit(self, frames: dict[str, np.ndarray | None]) -> None:
        for role, writer in self.writers.items():
            writer.submit(frames.get(role))

    def close(self) -> dict[str, Any]:
        """Close every file and return the `meta.json` video block."""
        files: dict[str, Any] = {}
        for role, writer in self.writers.items():
            stats = writer.close()
            files[role] = {
                "file": writer.path.name,
                "width": writer.width,
                "height": writer.height,
                **stats.to_json(),
                "error": writer.error,
            }
        return {"encoder": self.choice.to_json(), "cameras": files}

    def discard(self) -> None:
        for writer in self.writers.values():
            writer.close()
            writer.path.unlink(missing_ok=True)
