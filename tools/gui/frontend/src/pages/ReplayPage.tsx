import { useEffect, useState } from "react";
import { ApiError, api, previewUrl } from "../api";
import { useSeries, useTicker } from "../hooks";
import { SeriesPlot } from "../components/SeriesPlot";
import { Banner, Card, Empty, Field, Pill, Segmented, Stat } from "../components/ui";
import { DEFAULT_RIG, RigControls, rigBody, rigReady, type RigState } from "../components/RigControls";
import type { AppConfig, EpisodeMeta, ReplayMode, Snapshot } from "../types";

const MODE_HELP: Record<ReplayMode, string> = {
  joint: "把录下的关节指令原样发回去。回答「机械臂能不能复现当时的动作」——偏差来自硬件、伺服或场景,不来自解算。",
  task: "把录下的 5D 任务位姿用当前的 IK 和 atlas 重新解一遍。回答「今天的控制器会不会做同样的事」——这是改解算后的回归测试。"
};

export function ReplayPage({
  config,
  snapshot,
  run
}: {
  config: AppConfig;
  snapshot: Snapshot | null;
  run: (action: () => Promise<unknown>) => Promise<void>;
}) {
  const [episodes, setEpisodes] = useState<EpisodeMeta[]>([]);
  const [episodeId, setEpisodeId] = useState("");
  const [rig, setRig] = useState<RigState>({ ...DEFAULT_RIG });
  const [mode, setMode] = useState<ReplayMode>("joint");
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .episodes()
      .then((list) => {
        setEpisodes(list);
        setEpisodeId((current) => (current && list.some((e) => e.id === current) ? current : list[0]?.id ?? ""));
      })
      .catch((cause) => setError(cause instanceof ApiError ? cause.message : String(cause)));
  }, [snapshot?.mode === "idle"]);

  const sessionMode = snapshot?.mode ?? "idle";
  const idle = sessionMode === "idle";
  const replaying = sessionMode === "replay";
  const status = snapshot?.replay;
  const blocked = rigReady(rig, config);
  const episode = episodes.find((e) => e.id === episodeId);

  const series = useSeries(replaying);
  const previewOn = replaying && (snapshot?.spec?.backend ?? "") === "mujoco";
  const tick = useTicker(120, previewOn);

  const progress = status && status.total > 0 ? (100 * status.step) / status.total : 0;

  return (
    <div className="grid cols-2" style={{ gridTemplateColumns: "360px minmax(0, 1fr)" }}>
      <div className="grid" style={{ alignContent: "start" }}>
        <Card title="回放设置">
          {error ? <Banner tone="error">{error}</Banner> : null}

          <Field label="回放哪一条">
            <select
              value={episodeId}
              disabled={!idle}
              onChange={(event) => setEpisodeId(event.target.value)}
            >
              {episodes.length === 0 ? <option value="">（没有录制）</option> : null}
              {episodes.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id} · {item.n_steps} 帧 · {item.task || item.name || "未标注"}
                </option>
              ))}
            </select>
          </Field>

          <Field label="模式" hint={MODE_HELP[mode]}>
            <Segmented<ReplayMode>
              value={mode}
              disabled={!idle}
              onChange={setMode}
              options={[
                { value: "joint", label: "joint 关节复现" },
                { value: "task", label: "task 重解算" }
              ]}
            />
          </Field>

          <Field
            label={`速度 ${speed.toFixed(2)}×`}
            hint="超过 1× 机械臂真的会更快;超过速度上限的帧会被限速,回放会落后于录制。"
          >
            <input
              type="range"
              min={0.25}
              max={2}
              step={0.05}
              value={speed}
              disabled={!idle}
              onChange={(event) => setSpeed(Number(event.target.value))}
            />
          </Field>

          <RigControls rig={rig} onChange={setRig} config={config} disabled={!idle} showSource={false} />
          {blocked && idle ? <Banner tone="warn">{blocked}</Banner> : null}

          <div className="row">
            <button
              className="btn primary"
              disabled={!idle || !episodeId || Boolean(blocked)}
              onClick={() => {
                if (rig.backend === "real" && !window.confirm(
                  "真机回放:机械臂会先走到该条的第一帧,然后跑完整条轨迹。\n\n工作区清空了吗?手在急停旁吗?"
                )) return;
                void run(() =>
                  api.startReplay({ ...rigBody(rig), episode_id: episodeId, mode, speed })
                );
              }}
            >
              开始回放
            </button>
            <button className="btn danger" disabled={idle} onClick={() => void run(api.stop)}>
              停止
            </button>
          </div>
          <div className="dim small" style={{ marginTop: 8 }}>
            开始前会做一次静态检查:关节顺序、超出当前限位的指令、该速度下会不会超过限速。
            检查出错会直接拒绝回放。
          </div>
        </Card>

        {episode ? (
          <Card title="该条录制于" padded={false}>
            <div className="stats">
              <Stat label="来源" value={episode.backend} small />
              <Stat label="输入" value={episode.source} small />
              <Stat label="帧数" value={episode.n_steps} small />
              <Stat label="时长" value={episode.duration_s.toFixed(1)} unit="s" small />
            </div>
            {episode.task ? <div className="body small muted">{episode.task}</div> : null}
          </Card>
        ) : null}
      </div>

      <div className="grid" style={{ alignContent: "start" }}>
        <Card title="回放状态" padded={false}>
          {status && (status.active || status.step > 0) ? (
            <>
              <div className="stats">
                <Stat label="阶段" value={<Pill tone={status.active ? "ok" : "neutral"} live={status.active}>{phaseLabel(status.phase)}</Pill>} small />
                <Stat label="进度" value={`${status.step} / ${status.total}`} small />
                <Stat label="模式" value={status.mode} small />
                <Stat label="速度" value={`${status.speed.toFixed(2)}×`} small />
              </div>
              <div className="body">
                <div className="progress" style={{ marginBottom: 12 }}>
                  <div className="fill" style={{ width: `${progress}%` }} />
                </div>
                {status.phase === "approach" ? (
                  <Banner tone="info">
                    正在走到第一帧,还差 {status.approach_remaining_deg.toFixed(1)}°(限速、无 IK)。
                  </Banner>
                ) : null}
                {status.issues.map((issue, i) => (
                  <Banner key={i} tone={issue.level === "error" ? "error" : "warn"}>
                    {issue.message}
                  </Banner>
                ))}
                {status.aborted_reason ? <Banner tone="warn">{status.aborted_reason}</Banner> : null}
                {!status.active && Object.keys(status.summary).length > 0 ? (
                  <>
                    <div className="plot-title">
                      <span>结果 · {status.completed ? "完整跑完" : "中途停止"}</span>
                    </div>
                    <div className="stats" style={{ marginTop: 8 }}>
                      <Stat label="指令偏差 p95" value={status.summary.command_deviation_p95_deg?.toFixed(3)} unit="°" small />
                      <Stat label="指令偏差 max" value={status.summary.command_deviation_max_deg?.toFixed(3)} unit="°" small />
                      <Stat label="跟踪误差 p95" value={status.summary.tracking_error_p95_deg?.toFixed(3)} unit="°" small />
                      <Stat label="任务位置误差 p95" value={status.summary.task_position_error_p95_mm?.toFixed(3)} unit="mm" small />
                      <Stat label="限速帧" value={((status.summary.rate_clamped_frac ?? 0) * 100).toFixed(1)} unit="%" small />
                      <Stat label="安全保持帧" value={((status.summary.safety_held_frac ?? 0) * 100).toFixed(1)} unit="%" small />
                    </div>
                    <div className="dim small" style={{ marginTop: 10 }}>
                      指令偏差是发出去的和录下来的差 —— 非零说明限幅生效了,或者 task 模式解到了别的构型。
                      跟踪误差是实测和录下来的差 —— 那是伺服的账。
                    </div>
                  </>
                ) : null}
              </div>
            </>
          ) : (
            <Empty>选一条录制,按开始回放。</Empty>
          )}
        </Card>

        {replaying || series.length > 0 ? (
          <div className="grid cols-2">
            <Card title="偏差">
              <SeriesPlot
                x={series.map((row) => row.t)}
                series={[
                  { label: "指令偏差", values: series.map((row) => row.deviation_deg) },
                  { label: "跟踪误差", values: series.map((row) => row.tracking_deg) }
                ]}
                unit="deg"
                height={120}
                yMin={0}
              />
              <SeriesPlot
                x={series.map((row) => row.t)}
                series={[{ label: "任务位置误差", values: series.map((row) => row.pos_err_mm) }]}
                unit="mm"
                height={100}
                yMin={0}
              />
            </Card>
            <Card title="仿真相机">
              {previewOn ? (
                <div className="preview">
                  <img src={previewUrl("third_person", 640, 480, tick)} alt="third_person 相机" />
                  <div className="tag">third_person</div>
                </div>
              ) : (
                <Empty>仅 mujoco backend 有仿真相机</Empty>
              )}
            </Card>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function phaseLabel(phase: string): string {
  return { idle: "空闲", approach: "接近首帧", playing: "回放中", homing: "归位", done: "结束" }[phase] ?? phase;
}
