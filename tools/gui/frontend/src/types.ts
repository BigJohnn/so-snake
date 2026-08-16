// The gateway's payloads, mirrored. Field names are snake_case here because
// they are snake_case on the wire; renaming them in a mapping layer would buy a
// naming convention and cost the ability to grep one name across both halves.

/** "held" is homed-and-energized: nothing is driving the arm, but torque is on
 *  and it is standing at the home pose. Only a stop releases it. */
export type Mode = "idle" | "teleop" | "replay" | "homing" | "held";
export type BackendKind = "mock" | "mujoco" | "real";
export type SourceKind = "scripted" | "pro";
export type ReplayMode = "joint" | "task";

export interface Flags {
  workspace_clamped: boolean;
  atlas_pitch_clamped: boolean;
  atlas_roll_infeasible: boolean;
  joint_limit_clamped: boolean;
  joint_rate_clamped: boolean;
  command_safety_held: boolean;
  ik_converged: boolean;
}

export interface Telemetry {
  index: number;
  t: number;
  task_target: number[];
  achieved_task_pose: number[];
  position: number[];
  commanded_joints_deg: number[];
  measured_joints_deg: number[];
  gripper_cmd_deg: number;
  clutch: boolean;
  loop_hz: number;
  ik_position_error_mm: number;
  ik_pitch_error_deg: number;
  ik_roll_error_deg: number;
  robot_mesh_min_z_m: number | null;
  safety_reason: string;
  flags: Flags;
}

export interface RecordingStatus {
  recording: boolean;
  id: string | null;
  name: string;
  task: string;
  steps: number;
  duration_s: number;
  aborted_reason: string;
}

/** The take that just finished.
 *
 * `pending` means it ended on its own (it ran out of frames) and is waiting for
 * the operator to keep or discard it. The episode is already on disk either way
 * -- discarding deletes it -- because a take held in memory until someone
 * approved it is a take a crash would lose.
 */
export interface LastTake {
  id: string;
  name: string;
  task: string;
  n_steps: number;
  duration_s: number;
  pending: boolean;
}

/** Where homing goes.
 *
 * `source` is "file" when a start pose was recorded (assets/so100_start_pose.json)
 * and "config" when homing falls back to `TeleopConfig.home_joints_deg` -- which
 * also happens when a recorded pose is unusable, and then `error` says why.
 */
export interface StartPoseStatus {
  source: "file" | "config";
  path: string;
  error: string;
  joints_deg: Record<string, number>;
  recorded_at?: string;
  recorded_from?: string;
  task_pose_xyz_m?: number[];
  in_workspace_box?: boolean;
}

/** The take batch: one episode's length, how many the operator means to
 *  collect, and how many are done in this session. */
export interface TakeStatus {
  steps_per_take: number;
  target_count: number;
  done_count: number;
  control_hz: number;
}

export interface ReplayIssue {
  level: "error" | "warning";
  message: string;
}

export interface ReplayStatus {
  active: boolean;
  phase: "idle" | "approach" | "playing" | "homing" | "done";
  episode_id: string;
  mode: string;
  speed: number;
  step: number;
  total: number;
  approach_remaining_deg: number;
  /** How far off the first frame the approach ended. A servo settles a degree
   *  or two out; only a large value means something was holding the arm. */
  approach_residual_deg: number;
  completed: boolean;
  aborted_reason: string;
  issues: ReplayIssue[];
  summary: Record<string, number>;
}

/** One serial port, as reported by /api/ports.
 *
 * `likely` is the gateway's judgement that this could be the arm: a known USB
 * bridge chip, or a device name shaped like a USB serial adapter. The macOS
 * Bluetooth and debug-console ports are never likely.
 */
export interface SerialPort {
  device: string;
  label: string;
  usb_id: string;
  known_as: string;
  likely: boolean;
}

/** `detected` is "" when the gateway could not tell, and `reason` says why. */
export interface PortScan {
  ports: SerialPort[];
  detected: string;
  reason: string;
}

export type CameraRole = "third_person" | "wrist";

/** One camera attached to this machine, as reported by /api/cameras.
 *
 * `name` is empty on macOS on purpose -- the OpenCV index cannot be mapped to a
 * device name there. `thumbnail` is what the operator identifies it by.
 */
export interface CameraDevice {
  index: number;
  /** What to send back as this camera's device: a stable path where the
   *  platform has one (Linux /dev/v4l/by-id), otherwise the bare index. */
  device: number | string;
  /** True when `device` survives a replug. False means the index is only good
   *  until the set of connected cameras changes. */
  stable: boolean;
  /** "usb" where the platform actually says so (Linux by-id); "" where it
   *  cannot be known without guessing (macOS). Never inferred. */
  bus: string;
  name: string;
  width: number;
  height: number;
  thumbnail: string;
  /** Variance of the Laplacian: the standard focus measure. */
  sharpness: number;
  /** Greyscale standard deviation. Near zero is a blank or covered lens. */
  contrast: number;
  /** Why this camera is hard to tell apart from another by its picture, or "".
   *  Advisory only -- the device is listed, selectable, and recordable. A wrist
   *  camera focused at gripper distance reads low here by design. */
  note: string;
}

/** Why the scan found what it found. Shown when something is missing, because
 *  an empty or short list has several very different causes. */
export interface CameraDiagnostics {
  platform: string;
  max_index: number;
  attempted: number;
  opened: number;
  readable: number;
  failures: { index: number; reason: string }[];
  /** Present but low on detail. Nothing refuses these; the list exists so the
   *  UI can name the thumbnail that looks like an empty slot. */
  hard_to_identify: { index: number; reason: string }[];
  permission_hint: string;
}

export interface CameraScan {
  devices: CameraDevice[];
  diagnostics: CameraDiagnostics;
  roles: string[];
}

/** Which cameras a session asked for, and which of them actually opened. */
export interface CameraStatus {
  roles: string[];
  devices: Record<string, number | string>;
  connected: string[];
}

export interface SpecStatus {
  backend: BackendKind;
  source: SourceKind;
  port: string;
  physical: boolean;
  max_relative_target_deg: number;
  cameras: Record<string, number | string>;
}

export interface EventItem {
  time: string;
  level: "info" | "warn" | "error";
  message: string;
}

export interface Snapshot {
  mode: Mode;
  error: string;
  uptime_s: number;
  spec: SpecStatus | null;
  connected: boolean;
  steps: number;
  latest: Telemetry | Record<string, never>;
  cameras: CameraStatus;
  recording: RecordingStatus;
  start_pose: StartPoseStatus;
  last_take: LastTake;
  takes: TakeStatus;
  replay: ReplayStatus;
  events: EventItem[];
}

export interface SeriesRow {
  t: number;
  pos_err_mm: number;
  loop_hz: number;
  clutch: boolean;
  target: number[];
  joints: number[];
  deviation_deg: number;
  tracking_deg: number;
}

export interface Availability {
  backends: Record<BackendKind, { available: boolean; reason: string }>;
  sources: Record<SourceKind, { available: boolean; reason: string }>;
  cameras: { available: boolean; reason: string; roles: CameraRole[] };
  joint_map_path: string;
  joint_map_present: boolean;
}

export interface AppConfig {
  arm: {
    joint_names: string[];
    gripper_joint: string;
    joint_limits_deg: Record<string, [number, number]>;
    urdf_path: string;
  };
  limits: {
    pos_min_m: number[];
    pos_max_m: number[];
    pitch_min_deg: number;
    pitch_max_deg: number;
    max_step_pos_m: number;
  };
  teleop: {
    control_hz: number;
    home_joints_deg: number[];
    max_joint_step_deg: number;
    min_robot_mesh_z_m: number;
    gripper_open_deg: number;
    gripper_closed_deg: number;
  };
  episode_root: string;
  availability: Availability;
  cameras: string[];
}

export interface EpisodeMeta {
  id: string;
  name: string;
  task: string;
  notes: string;
  created_at: string;
  format_version: number;
  backend: string;
  source: string;
  simulated: boolean;
  n_steps: number;
  duration_s: number;
  control_hz: number;
  joint_names: string[];
  summary: Record<string, number>;
  aborted_reason: string;
  /** Empty for an episode recorded without cameras. */
  video: {
    encoder?: { codec: string; reason: string; hardware: boolean };
    cameras?: Record<string, {
      file: string; width: number; height: number;
      written: number; dropped: number; stale: number; error: string;
    }>;
  };
}

export interface EpisodeDetail {
  meta: EpisodeMeta;
  size_bytes: number;
  columns: string[];
  series: {
    stride: number;
    t: number[];
    task_target: number[][];
    achieved_task_pose: number[][];
    commanded_joints_deg: number[][];
    measured_joints_deg: number[][];
    gripper_deg: number[];
    clutch: boolean[];
    pos_err_mm: number[];
    loop_hz: number[];
    workspace_clamped: boolean[];
    atlas_pitch_clamped: boolean[];
    joint_rate_clamped: boolean[];
    command_safety_held: boolean[];
  };
}

export interface RoadmapItem {
  key: string;
  title: string;
  status: "done" | "partial" | "todo";
  detail: string;
  module: string;
  evidence: string;
  blockers: string[];
}

export interface Roadmap {
  groups: { group: string; items: RoadmapItem[] }[];
  counts: Record<string, number>;
  total: number;
}

// ------------------------------------------------------------------ export

export type ActionSpace = "delta" | "absolute";

/** One task label in the store, and what exporting it would contribute. */
export interface TaskSummary {
  task: string;
  takes: number;
  steps: number;
  seconds: number;
}

/** Why one take was or was not included in an export. */
export interface ExportEpisodeReport {
  episode_id: string;
  task: string;
  n_steps: number;
  measured_fps: number;
  gripper_measured: boolean;
  included: boolean;
  reason: string;
  /** What the loop was configured for when this take was recorded. Below
   *  `measured_fps` means the loop missed its rate then, which is baked into
   *  the take — no amount of re-exporting raises it. */
  configured_hz: number;
}

export interface ExportReport {
  fps: number;
  action_space: string;
  n_episodes: number;
  n_frames: number;
  cancelled: boolean;
  dataset_path: string;
  episode_ids: string[];
  action_stats: Record<string, any>;
  replay_check: Record<string, any>;
  episodes: ExportEpisodeReport[];
  skipped: ExportEpisodeReport[];
}

/** What reading the written dataset back off disk proved.
 *
 * `ok` false with an empty `issues` cannot happen: the issues are what makes it
 * false. An export is not finished until this comes back ok -- everything it
 * checks looks like success at the moment of writing. */
export interface VerifyReport {
  repo_id: string;
  dataset_path: string;
  fps: number;
  action_space: string;
  n_episodes: number;
  n_frames: number;
  state_max_abs_error: number;
  action_max_abs_error: number;
  target_position_error_max_m: number;
  target_angle_error_max_rad: number;
  gripper_error_max_deg: number;
  timestamp_max_error_s: number;
  video_frames: Record<string, number>;
  issues: string[];
  ok: boolean;
}

export type ExportPhase =
  | "idle"
  | "exporting"
  | "verifying"
  | "done"
  | "failed"
  | "cancelled";

export interface ExportProgress {
  phase: ExportPhase;
  running: boolean;
  /** "export" or "verify". Both are long, both use this slot, and the UI
   *  needs to know which it is watching so a verify on one dataset does
   *  not get conflated with an export on another. */
  kind: string;
  repo_id: string;
  task: string;
  dataset_path: string;
  episodes_done: number;
  episodes_total: number;
  frames_done: number;
  current_episode: string;
  error: string;
  report: ExportReport | null;
  verify_report: VerifyReport | null;
  log: string[];
}

/** What the dry run answers: would this selection export, and as what. */
export interface ExportPlan {
  report: ExportReport;
  usable: number;
  config: Record<string, any>;
}

// ---------------------------------------------------------------- datasets

/** One dataset on disk. Manifest and verdict are separate because one is the
 *  dataset's own claim about itself and the other is what reading it back
 *  proved -- they can disagree, and when they do the verdict wins.
 *
 *  `ours` distinguishes a manifest written by this exporter (so source-fidelity
 *  is checkable) from a synthesized one (built from lerobot's info.json when
 *  only the parquet is around -- replay works, verify-with-source does not). */
export interface DatasetMeta {
  name: string;
  path: string;
  size_bytes: number;
  modified: number;
  /** `null` when the directory is not a LeRobotDataset at all. When set but
   *  `ours=false`, the manifest is synthesized from lerobot's `info.json` and
   *  has no source mapping. */
  manifest: DatasetManifest | null;
  /** True when `manifest` was written by so-snake's exporter. */
  ours: boolean;
  /** `null` when no `verify.json` has been written yet. `stale=true` when the
   *  directory's mtime has moved past the recorded verification -- the cached
   *  answer no longer matches the bytes on disk. */
  verdict: DatasetVerdict | null;
}

/** What the export wrote into `export.json`: the contract a dataset carries
 *  with it, independent of the GUI. `episode_ids[i]` is the source take for
 *  dataset episode `i` -- the only place that mapping lives. */
export interface DatasetManifest {
  repo_id: string;
  root: string;
  task: string | null;
  fps: number;
  action_space: string;
  cameras: string[];
  resolution: number[];
  n_episodes: number;
  n_frames: number;
  cancelled: boolean;
  episode_ids: string[];
  episode_root: string;
}

/** What reading the dataset back off disk proved. The same shape `verify`
 *  produces in the CLI, with two extra fields: `stale` is the GUI's way of
 *  saying "the cached answer is older than the data it is about", and
 * `skipped` lists checks that did not run (no manifest, no source store)
 *  so a green `ok` cannot be read as "verified against source takes". */
export interface DatasetVerdict {
  repo_id: string;
  dataset_path: string;
  fps: number;
  action_space: string;
  n_episodes: number;
  n_frames: number;
  state_max_abs_error: number;
  action_max_abs_error: number;
  target_position_error_max_m: number;
  target_angle_error_max_rad: number;
  gripper_error_max_deg: number;
  timestamp_max_error_s: number;
  video_frames: Record<string, number>;
  issues: string[];
  skipped: string[];
  ok: boolean;
  verified_at: number;
  verified_mtime: number;
  stale: boolean;
}
