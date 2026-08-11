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
