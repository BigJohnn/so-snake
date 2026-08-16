import { useCallback, useEffect, useRef, useState, type SyntheticEvent } from "react";
import { ApiError, api, episodeVideoUrl } from "../api";
import { ExportPanel } from "../components/ExportPanel";
import { SeriesPlot } from "../components/SeriesPlot";
import { Banner, Card, Empty, Field, Pill, Stat } from "../components/ui";
import type { EpisodeDetail, EpisodeMeta } from "../types";

/* The recordings page: triage for *raw* takes, not the exported datasets.
 *
 * The two live under different roots (`data/episodes/` and `data/lerobot/`)
 * and have different lifecycles, so they live on different pages too -- a
 * verdict that applies to an export (the parquet reads back, the targets
 * reconstruct) is not a verdict on a take (did the grasp work). This page
 * answers the latter; the new "训练集" page answers the former.
 *
 * What actually happens here: a session just produced fifteen takes, and the
 * operator has to decide which ones are worth training on. That decision is
 * made by *watching* a take -- did the grasp land, did the arm clip the bin --
 * and then keeping it, deleting it, or writing down what it shows. The plots
 * matter afterwards, when a take looks wrong and the question becomes why.
 *
 * So the layout follows the loop rather than the data model:
 *
 *   * the list is a column that stays put, so choosing a take never scrolls the
 *     take away; arrow keys walk it, which is how fifteen of them get reviewed;
 *   * the video is the largest thing on the page, because it is the evidence;
 *   * the verdict (delete) and the label (task) sit next to the video, where the
 *     judgement is actually formed -- not in a toolbar above a table;
 *   * deleting selects the next take, because the operator is working down a
 *     list, not managing files;
 *   * the plots keep the shared cursor and go underneath; the diagnostics that
 *     answer "why" fold away until asked for.
 *
 * The export panel sits at the bottom because export is the last step of
 * review. Moving it onto the datasets page would mean leaving the page where
 * the decision about *which* takes was made.
 *
 * The previous version was a wide table stacked on a two-column detail, and the
 * detail had three children in a two-column grid -- so the trajectory card, the
 * widest thing on the page, wrapped into the 320px sidebar column while the
 * video sat alone in the wide one.
 */

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

  /* Arrow keys walk the list. Reviewing a session means going through every
   * take in order, and reaching for the mouse between each one is the friction
   * that stops people from reviewing at all. Ignored while typing, so the task
   * field still behaves like a text field. */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      const step = event.key === "ArrowDown" || event.key === "j" ? 1 : event.key === "ArrowUp" || event.key === "k" ? -1 : 0;
      if (step === 0 || episodes.length === 0) return;
      event.preventDefault();
      const at = episodes.findIndex((e) => e.id === selected);
      const next = Math.min(episodes.length - 1, Math.max(0, (at < 0 ? 0 : at) + step));
      setSelected(episodes[next].id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [episodes, selected]);

  const remove = async (id: string) => {
    if (!window.confirm(`删除 ${id}?此操作不可撤销(含视频)。`)) return;
    // Land on the neighbour rather than back at the top: the operator is working
    // down a list, and a delete is a step in that walk, not a return to the start.
    const at = episodes.findIndex((e) => e.id === id);
    const neighbour = episodes[at + 1]?.id ?? episodes[at - 1]?.id ?? "";
    try {
      await api.deleteEpisode(id);
      setSelected(neighbour);
      await reload();
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  };

  const totalSeconds = episodes.reduce((sum, e) => sum + e.duration_s, 0);

  return (
    <>
      {error ? <Banner tone="error">{error}</Banner> : null}
      <div className="dataset">
        <aside className="card take-list">
          <header>
            <h2>录制</h2>
            <div className="spacer">
              <button className="btn small" onClick={() => void reload()}>
                刷新
              </button>
            </div>
          </header>
          <div className="take-list-meta">
            {episodes.length} 条 · 共 {formatDuration(totalSeconds)}
            {episodes.length > 1 ? <span className="dim"> · ↑↓ 切换</span> : null}
          </div>
          <div className="take-scroll">
            {            episodes.length === 0 ? (
              <Empty>还没有原始 take 。到「遥操作」页启动会话并按录制 —— 录出来的原始数据在这里审,导出的训练集在「训练集」页。</Empty>
            ) : (
              episodes.map((episode) => (
                <TakeRow
                  key={episode.id}
                  episode={episode}
                  selected={episode.id === selected}
                  onSelect={() => setSelected(episode.id)}
                />
              ))
            )}
          </div>
        </aside>

        <div className="grid" style={{ alignContent: "start", minWidth: 0 }}>
          {detail ? (
            <Detail
              detail={detail}
              joint={joint}
              onJoint={setJoint}
              onSaved={reload}
              onDelete={() => void remove(detail.meta.id)}
            />
          ) : (
            <Card title="预览">
              <Empty>{episodes.length ? "选一条看看" : "还没有可看的 take"}</Empty>
            </Card>
          )}
          {/* Last, under the take being judged: exporting is what happens after
              the review, and a button for it above the video would invite
              pressing it before the reviewing is done. */}
          {episodes.length ? <ExportPanel /> : null}
        </div>
      </div>
    </>
  );
}

/** One take in the list: what it is, then how big it is. Two lines, no columns.
 *
 * A table row makes every field equally prominent; here they are not. The task
 * is what the operator is looking for, and the rest is context they only need
 * once they have found it. */
function TakeRow({
  episode,
  selected,
  onSelect
}: {
  episode: EpisodeMeta;
  selected: boolean;
  onSelect: () => void;
}) {
  const hasVideo = Object.keys(episode.video?.cameras ?? {}).length > 0;
  return (
    <button className={`take${selected ? " selected" : ""}`} onClick={onSelect}>
      <div className="title">
        {episode.task || episode.name || <span className="dim">未标注</span>}
        {episode.aborted_reason ? <span className="flag" title={episode.aborted_reason}>⚠</span> : null}
      </div>
      <div className="meta mono">
        <span className={episode.simulated ? "accent" : "physical"}>{episode.backend}</span>
        <span>{episode.duration_s.toFixed(1)}s</span>
        <span>{episode.n_steps} 帧</span>
        {hasVideo ? <span title="有相机录像">▣</span> : null}
        <span className="when">{shortTime(episode.created_at)}</span>
      </div>
    </button>
  );
}

/** The episode's cameras, kept in step with the plots and with each other.
 *
 * Alignment is by **frame index, not by timestamp**. The recorder writes one
 * video frame per control step, so video frame i is row i -- but the file's
 * frame rate is the configured control_hz while the loop actually ran a little
 * slower, so video time and recorded time drift apart over a take. Index is the
 * coordinate both halves genuinely share.
 *
 * The first camera carries the controls and is the timekeeper; the others
 * follow it. Following means **its transport is mirrored onto them** -- play,
 * pause, seek, playback rate -- and only then is the index nudge a correction.
 *
 * That distinction is the whole of this component. It previously mirrored
 * nothing: the followers had no controls and nobody ever called `play()` on
 * them, so the only thing that ever moved them was the seek in the index
 * effect below, driven by `timeupdate`.
 *
 * ## Why the position is sampled on an animation frame, not from `timeupdate`
 *
 * `timeupdate` is specified to fire *at least* every 250 ms, and browsers take
 * that as licence to fire it at about 4 Hz to keep the CPU asleep. An index
 * derived from it therefore advances in quanta of `fps / 4` frames: at 30 fps
 * the counter read 1, 9, 17, 25 -- eight frames at a time -- and the plot
 * cursor jumped in the same steps, however smoothly the video itself played.
 *
 * `currentTime` is continuous; only the *event* is coarse. So the position is
 * sampled on `requestAnimationFrame` while playing, and `onIndex` fires only
 * when the derived index actually changes -- at most once per video frame.
 * `SeriesPlot` memoises everything whose cost scales with sample count, so
 * moving the cursor at that rate rebuilds one `<line>` and not five
 * 1200-point paths.
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
  const raf = useRef(0);
  // The last index this component reported, so the rAF loop can fire `onIndex`
  // only on a real change. Kept in a ref rather than read from the `index`
  // prop: the loop must not be torn down and rebuilt on every frame it causes.
  const reported = useRef(-1);

  const followers = (): HTMLVideoElement[] =>
    cameras.slice(1).map((role) => refs.current[role]).filter((v): v is HTMLVideoElement => !!v);

  // Seeking is driven from outside (a click on a plot); playing is driven from
  // inside (the element's own clock). Only jump when the two disagree by more
  // than a couple of frames, or every position sample would fight the playback.
  useEffect(() => {
    const want = index / fps;
    for (const video of Object.values(refs.current)) {
      if (video && Math.abs(video.currentTime - want) > 2 / fps) video.currentTime = want;
    }
  }, [index, fps]);

  // Stop the sampler if the component goes away mid-play.
  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const sample = () => {
    const lead = refs.current[cameras[0]];
    if (!lead) return;
    const at = Math.round(lead.currentTime * fps);
    if (at !== reported.current) {
      reported.current = at;
      onIndex(at);
    }
    if (!lead.paused && !lead.ended) raf.current = requestAnimationFrame(sample);
  };

  const startSampling = () => {
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(sample);
  };
  const stopSampling = () => {
    cancelAnimationFrame(raf.current);
    sample(); // one last read, so a pause lands on the exact frame shown
  };

  const lead = {
    onPlay: () => {
      for (const video of followers()) void video.play().catch(() => undefined);
      startSampling();
    },
    onPause: stopSampling,
    onEnded: stopSampling,
    // A seek is mirrored the moment it lands rather than waiting for the next
    // sample: scrubbing the scene view should move the wrist view with it.
    onSeeked: (event: SyntheticEvent<HTMLVideoElement>) => {
      const t = event.currentTarget.currentTime;
      for (const video of followers()) video.currentTime = t;
      if (event.currentTarget.paused) sample();
    },
    onRateChange: (event: SyntheticEvent<HTMLVideoElement>) => {
      const rate = event.currentTarget.playbackRate;
      for (const video of followers()) video.playbackRate = rate;
    },
    // Kept as a backstop only. It is far too coarse to drive the cursor (see
    // the component docstring), but it costs nothing and covers the case where
    // the position moves without `play` firing.
    onTimeUpdate: () => {
      const video = refs.current[cameras[0]];
      if (video && video.paused) sample();
    }
  };

  return (
    <div className="videos" style={{ gridTemplateColumns: `repeat(${cameras.length}, minmax(0, 1fr))` }}>
      {cameras.map((role, i) => (
        <div className="preview" key={`${id}:${role}`}>
          <video
            ref={(node) => {
              refs.current[role] = node;
            }}
            src={episodeVideoUrl(id, role)}
            controls={i === 0}
            muted
            // The followers are played, so they need frames ahead of the
            // playhead. "metadata" was enough while they were only ever
            // seeked, and is not now.
            preload={i === 0 ? "metadata" : "auto"}
            {...(i === 0 ? lead : {})}
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
  onSaved,
  onDelete
}: {
  detail: EpisodeDetail;
  joint: number;
  onJoint: (index: number) => void;
  onSaved: () => Promise<void>;
  onDelete: () => void;
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
  const dirty = task !== detail.meta.task || notes !== detail.meta.notes;

  const save = async () => {
    await api.annotate(detail.meta.id, { task, notes });
    await onSaved();
    setSaved(true);
  };

  return (
    <>
      {/* The take itself: what it is, what to do about it, and the evidence.
          One card, because judging a take is one action, not three panels. */}
      <section className="card">
        <div className="take-head">
          <div className="ident">
            <span className="mono id">{detail.meta.id}</span>
            <Pill tone={detail.meta.simulated ? "accent" : "danger"}>{detail.meta.backend}</Pill>
            <Pill>{detail.meta.source}</Pill>
            {detail.meta.aborted_reason ? (
              <Pill tone="warn">{detail.meta.aborted_reason}</Pill>
            ) : null}
          </div>
          <div className="row">
            {saved && !dirty ? <Pill tone="ok">已保存</Pill> : null}
            <button className="btn small danger" onClick={onDelete} title="连同视频一起从磁盘删除">
              删除这条
            </button>
          </div>
        </div>

        <div className="take-label">
          <input
            value={task}
            placeholder="这条演示的是什么?(训练时的 language instruction)"
            onChange={(event) => setTask(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void save();
            }}
          />
          <button className="btn primary" disabled={!dirty} onClick={() => void save()}>
            保存标注
          </button>
        </div>

        <div className="take-facts mono">
          <span>{detail.meta.n_steps} 帧</span>
          <span>{detail.meta.duration_s.toFixed(1)} s</span>
          <span>{detail.meta.control_hz} Hz</span>
          <span>{(detail.size_bytes / 1024 / 1024).toFixed(1)} MB</span>
          <span className="dim">{detail.meta.created_at.replace("T", " ").slice(0, 19)}</span>
        </div>

        {videoCameras.length ? (
          <>
            <EpisodeVideos
              id={detail.meta.id}
              cameras={videoCameras}
              fps={fps}
              index={row}
              onIndex={setRow}
            />
            <div className="scrub">
              <button className="btn small" onClick={() => setRow(Math.max(0, row - 1))}>
                ◀
              </button>
              <button
                className="btn small"
                onClick={() => setRow(Math.min(detail.meta.n_steps - 1, row + 1))}
              >
                ▶
              </button>
              <input
                type="range"
                min={0}
                max={Math.max(0, detail.meta.n_steps - 1)}
                value={row}
                onChange={(event) => setRow(Number(event.target.value))}
              />
              <span className="mono small dim">
                第 {row + 1} / {detail.meta.n_steps} 帧 ·{" "}
                {(s.t[Math.min(cursor, s.t.length - 1)] ?? 0).toFixed(2)} s
              </span>
            </div>
          </>
        ) : (
          <div className="no-video">
            这条没有相机录像 —— 录制时没有指派相机。下面的曲线仍然可用。
          </div>
        )}
      </section>

      <Card title="轨迹" actions={<span className="dim small">点曲线可定位到那一帧</span>}>
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
          cursor={videoCameras.length ? cursor : null}
          onScrub={videoCameras.length ? scrub : undefined}
          x={s.t}
          series={[
            { label: "指令", values: column(s.commanded_joints_deg, joint) },
            { label: "实测", values: column(s.measured_joints_deg, joint) }
          ]}
          unit="deg"
          height={130}
        />
        <SeriesPlot
          cursor={videoCameras.length ? cursor : null}
          onScrub={videoCameras.length ? scrub : undefined}
          x={s.t}
          series={[{ label: "夹爪指令", values: s.gripper_deg }]}
          unit="deg"
          height={90}
        />
        <SeriesPlot
          cursor={videoCameras.length ? cursor : null}
          onScrub={videoCameras.length ? scrub : undefined}
          x={s.t}
          series={[{ label: "IK 位置误差", values: s.pos_err_mm }]}
          bands={[{ values: s.workspace_clamped }, { values: s.command_safety_held }]}
          unit="mm · 蓝底为限幅/保持"
          height={100}
          yMin={0}
        />
      </Card>

      {/* Answers "why was this take bad", which is a question that only comes up
          after the video has already been watched. Folded until then. */}
      <details className="card fold">
        <summary>指标与备注</summary>
        <div className="stats">
          {SUMMARY_KEYS.filter(([key]) => key in detail.meta.summary).map(([key, label, unit]) => (
            <Stat
              key={key}
              label={label}
              value={
                unit === "%"
                  ? (detail.meta.summary[key] * 100).toFixed(1)
                  : fmt(detail.meta.summary[key], 3)
              }
              unit={unit}
              small
            />
          ))}
        </div>
        <div className="body">
          <Field label="备注" hint="只改标签。帧数据和配置快照是录制时的事实,不可编辑。">
            <textarea rows={3} value={notes} onChange={(event) => setNotes(event.target.value)} />
          </Field>
          <div className="row">
            <button className="btn" disabled={!dirty} onClick={() => void save()}>
              保存标注
            </button>
            <span className="dim small">
              采样步长 1/{s.stride} · {detail.columns.length} 列
            </span>
          </div>
        </div>
      </details>
    </>
  );
}

const toDeg = (v: number) => (v * 180) / Math.PI;

/** HH:MM for a take from today, MM-DD for an older one.
 *
 * The list is one session's worth of takes nine times out of ten, and then the
 * clock is what tells them apart; the tenth time it is last week's, and then the
 * date is the only thing that matters.
 *
 * Parsed and formatted in local time rather than sliced out of the string:
 * `created_at` is UTC (`...T13:39:28+00:00`), while the take *id* beside it is
 * local (`ep_20260812_213928`), so slicing showed one take under two clocks
 * eight hours apart -- and rolled "today" over eight hours early. */
function shortTime(createdAt: string): string {
  const ms = Date.parse(createdAt);
  if (!Number.isFinite(ms)) return "—";
  const date = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  return sameDay
    ? `${pad(date.getHours())}:${pad(date.getMinutes())}`
    : `${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function formatDuration(seconds: number): string {
  if (seconds < 90) return `${seconds.toFixed(0)} s`;
  return `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`;
}

function fmt(value: number | undefined, digits: number): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}
