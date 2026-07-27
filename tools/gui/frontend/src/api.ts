import type {
  AppConfig,
  BackendKind,
  EpisodeDetail,
  EpisodeMeta,
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
}

export const api = {
  config: () => request<AppConfig>("/api/config"),
  roadmap: () => request<Roadmap>("/api/roadmap"),
  snapshot: () => request<Snapshot>("/api/snapshot"),
  series: (limit = 600) =>
    request<{ series: SeriesRow[] }>(`/api/series?limit=${limit}`).then((r) => r.series),

  episodes: () => request<{ episodes: EpisodeMeta[] }>("/api/episodes").then((r) => r.episodes),
  episode: (id: string) => request<EpisodeDetail>(`/api/episode?id=${encodeURIComponent(id)}`),
  annotate: (id: string, fields: { name?: string; task?: string; notes?: string }) =>
    post<EpisodeMeta>("/api/episode/annotate", { id, ...fields }),
  deleteEpisode: (id: string) => post<{ deleted: boolean }>("/api/episode/delete", { id }),

  startSession: (rig: RigBody) => post<Snapshot>("/api/session/start", rig),
  stop: () => post<Snapshot>("/api/session/stop"),
  home: (rig: RigBody) => post<Snapshot>("/api/session/home", rig),

  startRecording: (fields: { name?: string; task?: string; notes?: string }) =>
    post<Snapshot>("/api/record/start", fields),
  stopRecording: (keep: boolean) => post<Snapshot>("/api/record/stop", { keep }),

  startReplay: (body: RigBody & { episode_id: string; mode: ReplayMode; speed: number }) =>
    post<Snapshot>("/api/replay/start", body)
};

export const previewUrl = (camera: string, width: number, height: number, nonce: number) =>
  `/api/preview.png?camera=${camera}&w=${width}&h=${height}&n=${nonce}`;
