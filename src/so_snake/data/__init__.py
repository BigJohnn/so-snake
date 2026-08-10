"""Episode recording and replay.

`episode` defines the on-disk contract, `recorder` writes it from a live teleop
loop, `store` is the library over a directory of them, and `replay` drives an
arm from one -- either exactly as recorded, or through today's IK, which is how
a solver change gets regression-tested against real operator input.
"""

from .episode import (
    COLUMN_NAMES,
    DEFAULT_EPISODE_ROOT,
    FORMAT_VERSION,
    Episode,
    EpisodeMeta,
    encode_frames,
    read_episode,
    read_meta,
    write_episode,
)
from .recorder import EpisodeRecorder, config_snapshot
from .replay import (
    REPLAY_MODES,
    EpisodeReplayer,
    Issue,
    ReplayConfig,
    ReplayReport,
    ReplayStep,
    inspect_episode,
)
from .store import EpisodeStore
from .video import (
    EncoderChoice,
    VideoConfig,
    VideoSet,
    VideoStats,
    VideoWriter,
    probe_encoder,
    select_encoder,
)

__all__ = [
    "COLUMN_NAMES",
    "DEFAULT_EPISODE_ROOT",
    "FORMAT_VERSION",
    "REPLAY_MODES",
    "EncoderChoice",
    "Episode",
    "EpisodeMeta",
    "EpisodeRecorder",
    "EpisodeReplayer",
    "EpisodeStore",
    "Issue",
    "ReplayConfig",
    "ReplayReport",
    "ReplayStep",
    "VideoConfig",
    "VideoSet",
    "VideoStats",
    "VideoWriter",
    "config_snapshot",
    "encode_frames",
    "inspect_episode",
    "probe_encoder",
    "read_episode",
    "read_meta",
    "select_encoder",
    "write_episode",
]
