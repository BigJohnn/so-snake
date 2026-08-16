import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "./api";
import { useSnapshot } from "./hooks";
import { Banner, Pill } from "./components/ui";
import { TeleopPage } from "./pages/TeleopPage";
import { EpisodesPage } from "./pages/EpisodesPage";
import { DatasetsPage } from "./pages/DatasetsPage";
import { ReplayPage } from "./pages/ReplayPage";
import { RoadmapPage } from "./pages/RoadmapPage";
import type { AppConfig, Mode } from "./types";

// The four pages are the four artifacts an operator juggles, not four ways to
// view the same thing:
//   * "录制" -- raw takes under data/episodes/, the review loop on what was
//               recorded, ending in the export panel.
//   * "训练集" -- exported LeRobot datasets under data/lerobot/, the library of
//               what was produced, with verify and replay.
//   * "回放" -- replay a raw take on the arm; the question is about the
//               recording itself, so it sits with the recordings, not with
//               the exports.
// "进度" is the roadmap; it does not fit the artifact taxonomy but is the
// page the operator reaches for when stuck.
const PAGES = [
  { key: "teleop", label: "遥操作" },
  { key: "episodes", label: "录制" },
  { key: "datasets", label: "训练集" },
  { key: "replay", label: "回放" },
  { key: "roadmap", label: "进度" }
] as const;

type PageKey = (typeof PAGES)[number]["key"];

const MODE_LABEL: Record<Mode, string> = {
  idle: "空闲",
  teleop: "遥操作中",
  replay: "回放中",
  homing: "归位中",
  held: "已归位 · 保持力矩"
};

export default function App() {
  const [page, setPage] = useState<PageKey>("teleop");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [configError, setConfigError] = useState("");
  const [actionError, setActionError] = useState("");
  const { snapshot, error: pollError, refresh } = useSnapshot();

  useEffect(() => {
    api
      .config()
      .then(setConfig)
      .catch((cause) => setConfigError(cause instanceof ApiError ? cause.message : String(cause)));
  }, []);

  /* Every mutation funnels through here so that one place decides what a failure
   * looks like and every button gets a fresh snapshot afterwards -- the UI never
   * has to infer what its own click did. */
  const run = useCallback(
    async (action: () => Promise<unknown>) => {
      setActionError("");
      try {
        await action();
      } catch (cause) {
        setActionError(cause instanceof ApiError ? cause.message : String(cause));
      } finally {
        refresh();
      }
    },
    [refresh]
  );

  const mode = snapshot?.mode ?? "idle";
  const physical = Boolean(snapshot?.spec?.physical) && mode !== "idle";
  const busy = mode !== "idle";

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          so-snake<span>SO-ARM100 遥操作与数据采集</span>
        </div>
        <nav className="nav">
          {PAGES.map((item) => (
            <button
              key={item.key}
              className={page === item.key ? "active" : ""}
              onClick={() => setPage(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      <div className={`statusbar${physical ? " physical" : ""}`}>
        <Pill tone={busy ? "ok" : "neutral"} live={busy}>
          {MODE_LABEL[mode]}
        </Pill>
        {snapshot?.spec ? (
          <>
            <Pill tone={snapshot.spec.physical ? "danger" : "accent"}>
              {snapshot.spec.backend}
              {snapshot.spec.physical ? " · 真机" : ""}
            </Pill>
            <Pill>{snapshot.spec.source}</Pill>
          </>
        ) : null}
        {snapshot?.recording.recording ? (
          <Pill tone="danger" live>
            REC {snapshot.recording.steps} 帧 · {snapshot.recording.duration_s.toFixed(1)} s
          </Pill>
        ) : null}
        {busy ? <span className="dim">{snapshot?.steps ?? 0} 步</span> : null}
        {physical ? (
          <strong style={{ color: "var(--danger)" }}>真机通电中 —— 手放急停旁</strong>
        ) : null}
        <div style={{ marginLeft: "auto" }}>
          <button className="btn danger" disabled={!busy} onClick={() => void run(api.stop)}>
            停止 (全部)
          </button>
        </div>
      </div>

      <main className="page">
        {configError ? <Banner tone="error">读取配置失败: {configError}</Banner> : null}
        {pollError ? <Banner tone="error">网关连接失败: {pollError}</Banner> : null}
        {actionError ? <Banner tone="error">{actionError}</Banner> : null}
        {snapshot?.error ? <Banner tone="error">后台任务错误: {snapshot.error}</Banner> : null}

        {!config ? (
          <div className="empty">正在连接网关 ...</div>
        ) : page === "teleop" ? (
          <TeleopPage config={config} snapshot={snapshot} run={run} />
        ) : page === "episodes" ? (
          <EpisodesPage />
        ) : page === "datasets" ? (
          <DatasetsPage config={config} snapshot={snapshot} run={run} />
        ) : page === "replay" ? (
          <ReplayPage config={config} snapshot={snapshot} run={run} />
        ) : (
          <RoadmapPage />
        )}
      </main>
    </div>
  );
}
