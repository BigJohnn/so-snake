import type {
  AppConfig,
  BackendKind,
  CameraDevice,
  EpisodeDetail,
  EpisodeMeta,
  PortScan,
  ReplayMode,
  Roadmap,
  SeriesRow,
  Snapshot,
  SourceKind
} from "./types";

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
  // controls and never polled.
  cameras: () =>
    request<{ devices: CameraDevice[]; roles: string[] }>("/api/cameras").then((r) => r.devices),

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
    post<Snapshot>("/api/replay/start", body)
};

export const episodeVideoUrl = (id: string, camera: string) =>
  `/api/episode/video?id=${encodeURIComponent(id)}&camera=${encodeURIComponent(camera)}`;

export const previewUrl = (camera: string, width: number, height: number, nonce: number) =>
  `/api/preview.png?camera=${camera}&w=${width}&h=${height}&n=${nonce}`;
