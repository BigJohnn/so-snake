import type {
  ActionSpace,
  AppConfig,
  BackendKind,
  CameraScan,
  DatasetMeta,
  EpisodeDetail,
  EpisodeMeta,
  ExportPlan,
  ExportProgress,
  PortScan,
  ReplayMode,
  Roadmap,
  SeriesRow,
  Snapshot,
  SourceKind,
  TaskSummary
} from "./types";
import type { ModelMeta, TrainingStatus } from "./types";

// The gateway answers failures with `{"error": "..."}` and a 4xx/5xx, so the
// message the operator sees is the one the robot code actually raised --
// "cannot start a replay: the arm is busy (teleop)" rather than "request
// failed". Everything below funnels through here to keep that true.
export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (cause) {
    throw new ApiError(0, `gateway unreachable (${String(cause)})`);
  }
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new ApiError(response.status, text.slice(0, 200));
    }
  }
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload
        ? String((payload as { error: unknown }).error)
        : `HTTP ${response.status}`;
    throw new ApiError(response.status, message);
  }
  return payload as T;
}

function post<T>(path: string, body: unknown = {}): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

export interface RigBody {
  backend: BackendKind;
  source?: SourceKind;
  port?: string;
  robot_id?: string;
  max_relative_target_deg?: number;
  amplitude?: number;
  rotation_amplitude?: number;
  /** Role -> device index or path. Omitted roles are simply not opened. */
  cameras?: Record<string, number | string>;
}

export const api = {
  config: () => request<AppConfig>("/api/config"),
  roadmap: () => request<Roadmap>("/api/roadmap"),
  snapshot: () => request<Snapshot>("/api/snapshot"),
  series: (limit = 600) =>
    request<{ series: SeriesRow[] }>(`/api/series?limit=${limit}`).then((r) => r.series),

  // Slow (opens every device), so this is called on demand from the rig
  // controls and never polled. The diagnostics come back with it: an empty or
  // short list has several very different causes, and the operator cannot tell
  // them apart from the list alone.
  cameras: () => request<CameraScan>("/api/cameras"),

  // Cheap, unlike the camera scan: this reads the OS device list and opens
  // nothing, so it is safe even while the arm is connected.
  ports: () => request<PortScan>("/api/ports"),

  episodes: () => request<{ episodes: EpisodeMeta[] }>("/api/episodes").then((r) => r.episodes),
  episode: (id: string) => request<EpisodeDetail>(`/api/episode?id=${encodeURIComponent(id)}`),
  annotate: (id: string, fields: { name?: string; task?: string; notes?: string }) =>
    post<EpisodeMeta>("/api/episode/annotate", { id, ...fields }),
  deleteEpisode: (id: string) => post<{ deleted: boolean }>("/api/episode/delete", { id }),

  startSession: (rig: RigBody) => post<Snapshot>("/api/session/start", rig),
  stop: () => post<Snapshot>("/api/session/stop"),
  home: (rig: RigBody) => post<Snapshot>("/api/session/home", rig),

  // `steps` is the take's length in control frames (0 = until stopped);
  // `target_count` is how many takes the batch is for, and is only ever shown
  // back to the operator -- the gateway does not stop anything when it is hit.
  // Records the arm's current joints as the pose homing returns to. Meant to be
  // pressed mid-teleop, so it is a POST like every other mutation and comes back
  // with a fresh snapshot.
  captureStartPose: () => post<Snapshot>("/api/start-pose/capture"),

  startRecording: (fields: {
    name?: string;
    task?: string;
    notes?: string;
    steps?: number;
    target_count?: number;
  }) => post<Snapshot>("/api/record/start", fields),
  stopRecording: (keep: boolean) => post<Snapshot>("/api/record/stop", { keep }),
  // The verdict on a take that ran out of frames on its own. `false` deletes
  // the episode that is already on disk.
  decideLastTake: (keep: boolean) => post<Snapshot>("/api/record/decide", { keep }),

  startReplay: (body: RigBody & { episode_id: string; mode: ReplayMode; speed: number }) =>
    post<Snapshot>("/api/replay/start", body),

  // ---------------------------------------------------------------- export

  /** Task labels in the store, with what each would contribute to a dataset. */
  exportTasks: () =>
    request<{ tasks: TaskSummary[]; dataset_root: string }>("/api/export/tasks"),

  /** The dry run: screens the selection and reports, writing nothing. Cheap
   *  enough to be synchronous, which is the point -- a pre-flight the operator
   *  has to wait on a poll for is a pre-flight they will skip. */
  exportPlan: (body: ExportBody) => post<ExportPlan>("/api/export/plan", body),

  /** Starts the background job. Refused while the arm is being driven. */
  startExport: (body: ExportBody) => post<ExportProgress>("/api/export/start", body),
  exportStatus: () => request<ExportProgress>("/api/export/status"),
  cancelExport: () => post<ExportProgress>("/api/export/cancel"),

  // -------------------------------------------------------------- datasets

  /** The library of exported datasets. Lives under a separate path from the
   *  raw takes (`data/lerobot/` vs `data/episodes/`), which is why it is its
   *  own endpoint rather than a filter on `/api/episodes`. */
  datasets: () =>
    request<{ datasets: DatasetMeta[]; root: string }>("/api/export/datasets"),

  /** Re-run the read-back check on an already-written dataset. Long-running:
   *  it decodes every video, so the answer comes back through the existing
   *  `/api/export/status` poll. */
  startVerify: (dataset: string) =>
    post<ExportProgress>("/api/export/verify", { dataset }),

  /** Replay a dataset episode onto the arm, through the same safety layer a
   *  recorded take is replayed through. `mode` is fixed to `task` server-side:
   *  the dataset carries no joint stream on purpose. */
  startReplayDataset: (body: DatasetReplayBody) =>
    post<Snapshot>("/api/replay/dataset", body),

  // ------------------------------------------------------------ train/rollout
  models: () => request<{ models: ModelMeta[]; root: string }>("/api/models"),
  trainingStatus: () => request<TrainingStatus>("/api/training/status"),
  startTraining: (body: {
    dataset: string; policy: "act" | "pi05"; name: string; device: string; steps: number; batch_size: number; base_model?: string;
    wandb?: { enabled: boolean; project?: string; entity?: string; notes?: string; mode?: string };
    autodl?: { enabled: boolean; host?: string; user?: string; port?: string; root?: string; python?: string };
    pi_mode?: "expert" | "full";
  }) => post<TrainingStatus>("/api/training/start", body),
  stopTraining: () => post<TrainingStatus>("/api/training/stop"),
  deleteModels: (names: string[]) => post<{ deleted: string[] }>("/api/models/delete", { names }),
  startPolicyRollout: (body: {
    checkpoint: string; task: string; action_space: ActionSpace; backend: BackendKind; port?: string;
    cameras?: Record<string, number | string>; steps: number; device?: string;
    max_relative_target_deg?: number; confirm_real?: boolean;
  }) => post<TrainingStatus>("/api/rollout/start", body)
};

export interface DatasetReplayBody {
  dataset: string;
  episode_index?: number;
  backend?: BackendKind;
  source?: SourceKind;
  port?: string;
  robot_id?: string;
  max_relative_target_deg?: number;
  /** Playback rate multiplier. `speed <= 0` is rejected by the gateway. */
  speed?: number;
  /** When true, runs as fast as the safety layer allows; off by default
   *  because dataset replay is a thing people watch. */
  realtime?: boolean;
  /** Skip the MuJoCo mesh-clearance guard. Off by default. */
  check_clearance?: boolean;
}

export interface ExportBody {
  repo_id: string;
  task?: string | null;
  episode_ids?: string[];
  action_space?: ActionSpace;
  cameras?: string[];
  resolution?: [number, number];
  fps?: number | null;
  /** Normalised role -> [x, y, width, height] crop. */
  roi?: Record<string, [number, number, number, number]>;
  include_aborted?: boolean;
  /** Read the dataset back after writing and check it replays. Defaults true
   *  server-side, and there is no good reason to turn it off. */
  verify?: boolean;
  /** Wipe the target directory before writing. Destructive: the previous
   *  parquet, videos and manifest are deleted. Off by default -- the server
   *  refuses with a 409 if the directory already exists, so an unrelated
   *  dataset cannot be overwritten by mistake. */
  overwrite?: boolean;
}

export const episodeVideoUrl = (id: string, camera: string) =>
  `/api/episode/video?id=${encodeURIComponent(id)}&camera=${encodeURIComponent(camera)}`;

export const previewUrl = (camera: string, width: number, height: number, nonce: number) =>
  `/api/preview.png?camera=${camera}&w=${width}&h=${height}&n=${nonce}`;
