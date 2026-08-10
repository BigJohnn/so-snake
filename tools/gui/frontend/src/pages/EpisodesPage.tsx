import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, episodeVideoUrl } from "../api";
import { SeriesPlot } from "../components/SeriesPlot";
import { Banner, Card, Empty, Field, Pill, Stat } from "../components/ui";
import type { EpisodeDetail, EpisodeMeta } from "../types";

const SUMMARY_KEYS: [string, string, string][] = [
  ["ik_pos_err_p95_mm", "IK 位置误差 p95", "mm"],
  ["ik_pitch_err_p95_deg", "pitch 误差 p95", "°"],
  ["loop_hz_median", "回路频率中位", "Hz"],
  ["ik_solver_converged_frac", "解算收敛", "%"],
  ["workspace_clamped_frac", "工作区限幅", "%"],
  ["atlas_pitch_clamped_frac", "atlas 限幅", "%"],
  ["joint_rate_clamped_frac", "关节限速", "%"]
];

export function EpisodesPage() {
  const [episodes, setEpisodes] = useState<EpisodeMeta[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [detail, setDetail] = useState<EpisodeDetail | null>(null);
  const [error, setError] = useState("");
  const [joint, setJoint] = useState(0);

  const reload = useCallback(async () => {
    try {
      const list = await api.episodes();
      setEpisodes(list);
      setError("");
      setSelected((current) => (current && list.some((e) => e.id === current) ? current : list[0]?.id ?? ""));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    let alive = true;
    api
      .episode(selected)
      .then((value) => alive && setDetail(value))
      .catch((cause) => alive && setError(cause instanceof ApiError ? cause.message : String(cause)));
    return () => {
      alive = false;
    };
  }, [selected]);

  const remove = async (id: string) => {
    if (!window.confirm(`删除 ${id}?此操作不可撤销。`)) return;
    try {
      await api.deleteEpisode(id);
      await reload();
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  };

  return (
    <div className="grid">
      {error ? <Banner tone="error">{error}</Banner> : null}

      <Card
        title={`数据集 (${episodes.length} 条)`}
        padded={false}
        actions={
          <button className="btn small" onClick={() => void reload()}>
            刷新
          </button>
        }
      >
        {episodes.length === 0 ? (
          <Empty>还没有录制。到「遥操作 / 录制」页启动会话并按录制。</Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>id</th>
                <th>任务 / 名称</th>
                <th>来源</th>
                <th className="num">帧数</th>
                <th className="num">时长</th>
                <th className="num">IK p95</th>
                <th>录制于</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {episodes.map((episode) => (
                <tr
                  key={episode.id}
                  className={episode.id === selected ? "selected" : ""}
                  onClick={() => setSelected(episode.id)}
                >
                  <td className="mono small">{episode.id}</td>
                  <td>
                    {episode.task || <span className="dim">未标注任务</span>}
                    {episode.name ? <span className="dim small"> · {episode.name}</span> : null}
                    {episode.aborted_reason ? (
                      <div>
                        <Pill tone="warn">{episode.aborted_reason}</Pill>
                      </div>
                    ) : null}
                  </td>
                  <td>
                    <Pill tone={episode.simulated ? "accent" : "danger"}>{episode.backend}</Pill>{" "}
                    <span className="dim small">{episode.source}</span>
                  </td>
                  <td className="num">{episode.n_steps}</td>
                  <td className="num">{episode.duration_s.toFixed(1)}s</td>
                  <td className="num">{fmt(episode.summary.ik_pos_err_p95_mm, 3)}</td>
                  <td className="mono small dim">{episode.created_at.replace("T", " ").slice(0, 19)}</td>
                  <td>
                    <button
                      className="btn small danger"
                      onClick={(event) => {
                        event.stopPropagation();
                        void remove(episode.id);
                      }}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {detail ? <Detail detail={detail} joint={joint} onJoint={setJoint} onSaved={reload} /> : null}
    </div>
  );
}

/** The episode's cameras, kept in step with the plots.
 *
 * Alignment is by **frame index, not by timestamp**. The recorder writes one
 * video frame per control step, so video frame i is row i -- but the file's
 * frame rate is the configured control_hz while the loop actually ran a little
 * slower, so video time and recorded time drift apart over a take. Index is the
 * coordinate both halves genuinely share.
 *
 * Both cameras play from one clock: the first is the timekeeper and the second
 * is nudged back to it when they drift by more than a couple of frames, which
 * is what keeps a wrist view and a scene view showing the same instant.
 */
function EpisodeVideos({
  id,
  cameras,
  fps,
  index,
  onIndex
}: {
  id: string;
  cameras: string[];
  fps: number;
  index: number;
  onIndex: (i: number) => void;
}) {
  const refs = useRef<Record<string, HTMLVideoElement | null>>({});

  // Seeking is driven from outside (a click on a plot); playing is driven from
  // inside (the element's own clock). Only jump when the two disagree by more
  // than a couple of frames, or every timeupdate would fight the playback.
  useEffect(() => {
    const want = index / fps;
    for (const video of Object.values(refs.current)) {
      if (video && Math.abs(video.currentTime - want) > 2 / fps) video.currentTime = want;
    }
  }, [index, fps]);

  return (
    <div className={`grid${cameras.length > 1 ? " cols-2" : ""}`}>
      {cameras.map((role, i) => (
        <div className="preview" key={role}>
          <video
            ref={(node) => {
              refs.current[role] = node;
            }}
            src={episodeVideoUrl(id, role)}
            controls={i === 0}
            muted
            preload="metadata"
            onTimeUpdate={
              i === 0
                ? (event) => onIndex(Math.round(event.currentTarget.currentTime * fps))
                : undefined
            }
          />
          <div className="tag">{role}</div>
        </div>
      ))}
    </div>
  );
}

function Detail({
  detail,
  joint,
  onJoint,
  onSaved
}: {
  detail: EpisodeDetail;
  joint: number;
  onJoint: (index: number) => void;
  onSaved: () => Promise<void>;
}) {
  const [task, setTask] = useState(detail.meta.task);
  const [notes, setNotes] = useState(detail.meta.notes);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setTask(detail.meta.task);
    setNotes(detail.meta.notes);
    setSaved(false);
  }, [detail.meta.id, detail.meta.task, detail.meta.notes]);

  const s = detail.series;
  const jointNames = detail.meta.joint_names.filter((n) => n !== "gripper");

  // Row index into the full recording. The plots are decimated by `stride`, so
  // a plot sample maps to a row by multiplying and back by dividing; keeping
  // the shared cursor in row space means the video does not inherit the plots'
  // resolution.
  const [row, setRow] = useState(0);
  useEffect(() => setRow(0), [detail.meta.id]);
  const videoCameras = Object.keys(detail.meta.video?.cameras ?? {});
  const fps = detail.meta.control_hz || 30;
  const cursor = Math.round(row / Math.max(1, s.stride));
  const scrub = (i: number) => setRow(i * s.stride);
  const column = (rows: number[][], index: number) => rows.map((row) => row[index]);

  return (
    <div className="grid cols-2" style={{ gridTemplateColumns: "320px minmax(0, 1fr)" }}>
      <div className="grid" style={{ alignContent: "start" }}>
        <Card title="标注">
          <Field label="任务描述">
            <input value={task} onChange={(event) => setTask(event.target.value)} />
          </Field>
          <Field label="备注">
            <textarea rows={3} value={notes} onChange={(event) => setNotes(event.target.value)} />
          </Field>
          <div className="row">
            <button
              className="btn primary"
              onClick={async () => {
                await api.annotate(detail.meta.id, { task, notes });
                await onSaved();
                setSaved(true);
              }}
            >
              保存标注
            </button>
            {saved ? <Pill tone="ok">已保存</Pill> : null}
          </div>
          <div className="dim small" style={{ marginTop: 10 }}>
            只改标签。帧数据和配置快照是录制时的事实,不可编辑。
          </div>
        </Card>

        <Card title="录制条件" padded={false}>
          <div className="stats">
            <Stat label="帧数" value={detail.meta.n_steps} small />
            <Stat label="时长" value={detail.meta.duration_s.toFixed(1)} unit="s" small />
            <Stat label="控制频率" value={detail.meta.control_hz} unit="Hz" small />
            <Stat label="磁盘" value={(detail.size_bytes / 1024).toFixed(0)} unit="KB" small />
            <Stat label="列数" value={detail.columns.length} small />
            <Stat label="采样步长" value={`1/${s.stride}`} small />
          </div>
        </Card>

        <Card title="指标" padded={false}>
          <div className="stats">
            {SUMMARY_KEYS.filter(([key]) => key in detail.meta.summary).map(([key, label, unit]) => (
              <Stat
                key={key}
                label={label}
                value={unit === "%" ? (detail.meta.summary[key] * 100).toFixed(1) : fmt(detail.meta.summary[key], 3)}
                unit={unit}
                small
              />
            ))}
          </div>
        </Card>
      </div>

      {videoCameras.length ? (
        <Card
          title="相机"
          actions={
            <span className="dim small">
              第 {row + 1} / {detail.meta.n_steps} 帧 · t = {(s.t[Math.min(cursor, s.t.length - 1)] ?? 0).toFixed(2)} s
              {" · "}点击下方曲线可定位
            </span>
          }
        >
          <EpisodeVideos
            id={detail.meta.id}
            cameras={videoCameras}
            fps={fps}
            index={row}
            onIndex={setRow}
          />
        </Card>
      ) : null}

      <Card title={`轨迹 · ${detail.meta.id}`}>
        <SeriesPlot
          cursor={videoCameras.length ? cursor : null}
          onScrub={videoCameras.length ? scrub : undefined}
          x={s.t}
          series={[
            { label: "x", values: column(s.task_target, 0) },
            { label: "y", values: column(s.task_target, 1) },
            { label: "z", values: column(s.task_target, 2) }
          ]}
          bands={[{ values: s.clutch }]}
          unit="m · 蓝底为离合按下"
          height={130}
        />
        <SeriesPlot
          cursor={videoCameras.length ? cursor : null}
          onScrub={videoCameras.length ? scrub : undefined}
          x={s.t}
          series={[
            { label: "pitch", values: column(s.task_target, 3).map(toDeg) },
            { label: "roll", values: column(s.task_target, 4).map(toDeg) }
          ]}
          unit="deg"
          height={120}
        />

        <div className="row" style={{ margin: "6px 0" }}>
          <span className="dim small">关节</span>
          {jointNames.map((name, i) => (
            <button
              key={name}
              className={`btn small${joint === i ? " primary" : ""}`}
              onClick={() => onJoint(i)}
            >
              {name}
            </button>
          ))}
        </div>
        <SeriesPlot
          x={s.t}
          series={[
            { label: "指令", values: column(s.commanded_joints_deg, joint) },
            { label: "实测", values: column(s.measured_joints_deg, joint) }
          ]}
          unit="deg"
          height={130}
        />
        <SeriesPlot
          x={s.t}
          series={[{ label: "夹爪指令", values: s.gripper_deg }]}
          unit="deg"
          height={90}
        />
        <SeriesPlot
          x={s.t}
          series={[{ label: "IK 位置误差", values: s.pos_err_mm }]}
          bands={[{ values: s.workspace_clamped }, { values: s.command_safety_held }]}
          unit="mm · 蓝底为限幅/保持"
          height={100}
          yMin={0}
        />
      </Card>
    </div>
  );
}

const toDeg = (v: number) => (v * 180) / Math.PI;

function fmt(value: number | undefined, digits: number): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}
