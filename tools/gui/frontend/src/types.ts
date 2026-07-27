// The gateway's payloads, mirrored. Field names are snake_case here because
// they are snake_case on the wire; renaming them in a mapping layer would buy a
// naming convention and cost the ability to grep one name across both halves.

export type Mode = "idle" | "teleop" | "replay" | "homing";
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

export interface SpecStatus {
  backend: BackendKind;
  source: SourceKind;
  port: string;
  physical: boolean;
  max_relative_target_deg: number;
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
  recording: RecordingStatus;
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
