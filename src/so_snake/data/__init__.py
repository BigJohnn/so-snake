"""Episode recording and replay.

`episode` defines the on-disk contract, `recorder` writes it from a live teleop
loop, `store` is the library over a directory of them, `replay` drives an arm
from one -- either exactly as recorded, or through today's IK, which is how a
solver change gets regression-tested against real operator input -- and `export`
converts a selection of them into the `LeRobotDataset` a policy trains on.

`export` reaches for lerobot only inside the functions that need it, so this
package still imports on a machine that has nothing but numpy.
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
from .export import (
    ACTION_SPACES,
    MANIFEST_NAME,
    ExportConfig,
    ExportReport,
    VerifyReport,
    apply_action,
    build_state_action,
    crop_image,
    dataset_meta,
    export,
    format_report,
    format_verify,
    measured_fps,
    observed_task_pose,
    plan,
    read_manifest,
    replay_targets_from_state_action,
    sample_indices,
    validate_roi,
    verify,
    write_manifest,
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
    "ACTION_SPACES",
    "COLUMN_NAMES",
    "DEFAULT_EPISODE_ROOT",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "REPLAY_MODES",
    "EncoderChoice",
    "Episode",
    "EpisodeMeta",
    "EpisodeRecorder",
    "EpisodeReplayer",
    "EpisodeStore",
    "ExportConfig",
    "ExportReport",
    "Issue",
    "ReplayConfig",
    "ReplayReport",
    "ReplayStep",
    "VerifyReport",
    "VideoConfig",
    "VideoSet",
    "VideoStats",
    "VideoWriter",
    "apply_action",
    "build_state_action",
    "crop_image",
    "config_snapshot",
    "dataset_meta",
    "encode_frames",
    "export",
    "format_report",
    "format_verify",
    "inspect_episode",
    "measured_fps",
    "observed_task_pose",
    "plan",
    "probe_encoder",
    "read_episode",
    "read_manifest",
    "read_meta",
    "replay_targets_from_state_action",
    "sample_indices",
    "validate_roi",
    "select_encoder",
    "verify",
    "write_episode",
    "write_manifest",
]
