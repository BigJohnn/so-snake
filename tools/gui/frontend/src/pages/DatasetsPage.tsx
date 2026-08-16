import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api, type DatasetReplayBody } from "../api";
import {
  DEFAULT_RIG,
  RigControls,
  rigBody,
  rigReady,
  type RigState
} from "../components/RigControls";
import { Banner, Card, Empty, Field, Pill, Stat } from "../components/ui";
import type {
  AppConfig,
  DatasetMeta,
  ExportProgress,
  Snapshot
} from "../types";

/* The dataset library: every exported `LeRobotDataset` under `data/lerobot/`,
 * with the two things that make an export trustworthy: the manifest it claims
 * about itself and the read-back verdict that proves it.
 *
 * Two things distinguish this page from the recordings page, and both are
 * about keeping one question per page:
 *
 *   * **the recordings page** is the review loop on raw takes -- "did the take
 *     go well, should I keep it for training". It answers about the bytes
 *     under `data/episodes/`. The export panel sits at the bottom because
 *     export is the last step of that loop.
 *   * **this page** is the library of what the export produced -- the bytes
 *     under `data/lerobot/`. It answers two questions about them: "do the
 *     rows on disk match what the operator recorded?" (verify), and "will a
 *     policy trained here actually drive the arm?" (replay).
 *
 * Mixing them on one page invited reading a verdict that only applies to the
 * export as if it spoke about the take. Splitting them makes each page's
 * verdict mean what it says, and means the operator never has to ask which
 * artifact a green badge is about.
 *
 * The same physical action drives both pages' replay: the arm walks through a
 * recorded trajectory. The source differs -- a parquet on disk here, a recorded
 * episode in the store on the recordings page -- but the rig, the speed, the
 * safety layer and the verdict are the same shape, so this page borrows the
 * rig controls and the speed slider rather than re-inventing them.
 */

const POLL_MS = 1000;

export function DatasetsPage({
  config,
  snapshot,
  run
}: {
  config: AppConfig;
  snapshot: Snapshot | null;
  run: (action: () => Promise<unknown>) => Promise<void>;
}) {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState<ExportProgress | null>(null);

  const reload = useCallback(async () => {
    try {
      const payload = await api.datasets();
      setDatasets(payload.datasets);
      setError("");
      // Sticky on a name rather than a path: the operator picks datasets by
      // what they are called, and the path can move (e.g. after a re-export
      // into the same root) without the name changing.
      setSelected((current) =>
        current && payload.datasets.some((d) => d.name === current) ? current : payload.datasets[0]?.name ?? ""
      );
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  // Mirror the export-poller's pattern: only tick while something is running,
  // because a verification of a large dataset is minutes, and reloading the
  // library on every tick would re-walk the dataset root for no reason.
  // When a verify finishes, the page reloads once to pick up the cached
  // verdict that has just been written next to the dataset.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const status = await api.exportStatus();
        if (!cancelled) setProgress(status);
      } catch {
        /* a failed poll is not worth a banner; the next one will say */
      }
    };
    void tick();
    const timer = window.setInterval(() => {
      if (progress?.running) void tick();
    }, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [progress?.running]);

  // A verify that just finished is the cue to refresh the library: the verdict
  // is on disk by the time the phase flips out of "verifying", so the next
  // reload picks it up.
  useEffect(() => {
    if (progress && !progress.running && (progress.kind === "verify" || progress.kind === "export")) {
      void reload();
    }
  }, [progress?.running, progress?.phase, reload]);

  const meta = useMemo(() => datasets.find((d) => d.name === selected) ?? null, [datasets, selected]);
  // The verify that is *about* the dataset the operator is looking at, not
  // whatever happens to be running. An export on a different task will tick
  // the same progress feed but is not relevant to the verdict on this row.
  const runningVerify =
    progress && progress.running && progress.kind === "verify" && progress.repo_id === selected;

  const reVerify = async () => {
    if (!meta) return;
    try {
      setProgress(await api.startVerify(meta.path));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  };

  return (
    <>
      {error ? <Banner tone="error">{error}</Banner> : null}
      <div className="dataset">
        <aside className="card take-list">
          <header>
            <h2>训练集</h2>
            <div className="spacer">
              <button className="btn small" onClick={() => void reload()}>
                刷新
              </button>
            </div>
          </header>
          <div className="take-list-meta">
            {datasets.length} 个 · 写到 <span className="mono">{datasets[0]?.path?.split("/").slice(0, -1).join("/") ?? "—"}</span>
          </div>
          <div className="take-scroll">
            {datasets.length === 0 ? (
              <Empty>
                还没有训练集。到「录制」页审完一批原始 take,按导出 —— 导出的训练集出现在这里。
              </Empty>
            ) : (
              datasets.map((dataset) => (
                <DatasetRow
                  key={dataset.name}
                  dataset={dataset}
                  selected={dataset.name === selected}
                  onSelect={() => setSelected(dataset.name)}
                />
              ))
            )}
          </div>
        </aside>

        <div className="grid" style={{ alignContent: "start", minWidth: 0 }}>
          {meta ? (
            <DatasetDetail
              meta={meta}
              config={config}
              snapshot={snapshot}
              runningVerify={Boolean(runningVerify)}
              verifyProgress={runningVerify ? progress : null}
              onVerify={() => void reVerify()}
              run={run}
            />
          ) : (
            <Card title="预览">
              <Empty>{datasets.length ? "选一个训练集看看" : "还没有可看的训练集"}</Empty>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}

/** One row in the dataset library: name on top, the verdict's first sentence
 *  below. The verdict is the only sentence that matters about a dataset --
 *  "does it replay" -- and showing the size next to it gives equal weight to
 *  a thing nobody can act on. */
function DatasetRow({
  dataset,
  selected,
  onSelect
}: {
  dataset: DatasetMeta;
  selected: boolean;
  onSelect: () => void;
}) {
  const m = dataset.manifest;
  // "无 manifest" means: synthesized from lerobot's info.json only, no
  // source-take mapping. Replay still works; verify-with-source cannot.
  // The flag glyph reads "this dataset is real but partial".
  const partial = m !== null && !dataset.ours;
  return (
    <button
      className={`take dataset-row${selected ? " selected" : ""}`}
      onClick={onSelect}
      title={dataset.path}
    >
      <div className="title">
        {/* Glyph distinguishes dataset rows from take rows at a glance, even
            before the amber rail registers. ▣ is the recording/video glyph
            on the takes page; ▤ is the artefact glyph here. */}
        <span className="dim">▤</span>
        {m?.repo_id?.split("/").pop() || dataset.name || <span className="dim">未命名</span>}
        {partial ? (
          <span className="flag" title="没有 so-snake 的 export.json —— replay 可用,verify 只跑 round-trip">?</span>
        ) : null}
      </div>
      <div className="meta mono">
        <Pill tone={verdictBadge(dataset)}>{verdictLabel(dataset)}</Pill>
        {m ? (
          <>
            <span>{m.n_episodes} 条</span>
            <span>{m.fps} Hz</span>
          </>
        ) : (
          <span className="dim">无 manifest</span>
        )}
        {/* `modified` is epoch *seconds*, as Python hands it over; shortTime
            takes milliseconds. Without the conversion every dataset in the
            library was stamped 01-21, which is 1970 plus 20-odd days. */}
        <span className="when">{shortTime(dataset.modified * 1000)}</span>
      </div>
    </button>
  );
}

/** The selected dataset: what it claims, what was proved, and what to do with it.
 *
 * The parts sit in the order the operator works through them: identity first
 * (so they know they are looking at the right one), then the manifest's
 * claim about itself (what it was built from), then the verdict (the proof),
 * then replay (the action). Putting replay above the verdict would invite
 * pressing replay on a dataset that has just been shown to not replay.
 *
 * A dataset without our `export.json` (foreign, legacy, or wiped by hand) is
 * still listed and still replayable: the parquet is the source of truth for
 * replay, the manifest only adds source mapping. Verify still runs -- it just
 * marks source-fidelity as skipped rather than failing, and the verdict says
 * PARTIAL instead of OK. The "原始 take"折叠 is gated on `meta.ours` because
 * for a foreign dataset there is nothing to fold. */
function DatasetDetail({
  meta,
  config,
  snapshot,
  runningVerify,
  verifyProgress,
  onVerify,
  run
}: {
  meta: DatasetMeta;
  config: AppConfig;
  snapshot: Snapshot | null;
  runningVerify: boolean;
  verifyProgress: ExportProgress | null;
  onVerify: () => void;
  run: (action: () => Promise<unknown>) => Promise<void>;
}) {
  const manifest = meta.manifest;
  const verdict = meta.verdict;
  const sessionMode = snapshot?.mode ?? "idle";
  const idle = sessionMode === "idle";
  const held = sessionMode === "held";
  const replaying = sessionMode === "replay";
  const ready = idle || held;

  // Dataset replay is its own thing, with its own speed and (when the dataset
  // has more than one episode) its own episode picker. Resetting on dataset
  // change keeps the operator's speed choice from carrying over a switch.
  const [episodeIndex, setEpisodeIndex] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [rig, setRig] = useState<RigState>({ ...DEFAULT_RIG });
  useEffect(() => {
    setEpisodeIndex(0);
    setSpeed(1);
    setRig({ ...DEFAULT_RIG });
  }, [meta.name]);

  const blocked = rigReady(rig, config);
  // When the manifest is missing, the GUI synthesises one from info.json so
  // the episode picker still has a count. Trusting `manifest.n_episodes`
  // whether ours or synthesised is what keeps the picker from going blank on
  // a foreign dataset.
  const episodeCount = manifest?.n_episodes ?? 0;
  const validEpisode = episodeIndex >= 0 && episodeIndex < episodeCount;

  const replay = async () => {
    if (!manifest) return;
    const body: DatasetReplayBody = {
      // `meta.path`, which the gateway put there, and not anything off the
      // manifest: `export.json` records what the dataset was built *from*
      // (`episode_root`), never where it itself lives. The same value the
      // verify button sends.
      dataset: meta.path,
      episode_index: episodeIndex,
      ...rigBody(rig),
      speed,
      realtime: true,
      check_clearance: true
    };
    if (
      rig.backend === "real" &&
      !window.confirm(
        `真机回放数据集 ${manifest.repo_id} 第 ${episodeIndex} 条:\n机械臂会先走到第一帧,然后跑完整段轨迹。\n\n工作区清空了吗?手在急停旁吗?`
      )
    ) {
      return;
    }
    await run(() => api.startReplayDataset(body));
  };

  return (
    <>
      <section className="card">
        <div className="take-head">
          <div className="ident">
            <span className="mono id">{manifest?.repo_id ?? meta.name}</span>
            <Pill tone={verdictBadge(meta)}>{verdictLabel(meta)}</Pill>
            {manifest ? (
              <>
                <Pill tone="accent">{manifest.action_space}</Pill>
                <Pill>{manifest.fps} Hz</Pill>
              </>
            ) : null}
            {/* `ours=false` means we did not write this manifest -- it is a
                synthesis from lerobot's info.json. The pill is what the eye
                lands on before reading the verdict, and it is the difference
                between "round-trip ok" and "fully verified" that an operator
                most needs to see at a glance. */}
            {manifest && !meta.ours ? <Pill tone="warn">无 manifest</Pill> : null}
          </div>
          <div className="row">
            <button
              className="btn small"
              disabled={runningVerify || !validEpisode || replaying}
              onClick={onVerify}
              title={
                meta.ours
                  ? "读回 parquet + 视频,核对 targets 能否还原,再跟源 take 比对"
                  : "读回 parquet + 视频,核对 round-trip;没有 manifest,跳过跟源 take 比对"
              }
            >
              {runningVerify ? "校验中…" : verdict ? "重新校验" : "校验"}
            </button>
          </div>
        </div>

        <div className="take-facts mono">
          {manifest ? (
            <>
              <span>{manifest.n_episodes} 条 / {manifest.n_frames} 帧</span>
              <span>{(meta.size_bytes / 1024 / 1024).toFixed(1)} MB</span>
              <span>{manifest.task || "(所有技能)"}</span>
              <span className="dim">{meta.path}</span>
            </>
          ) : (
            <span className="dim">目录里既没有 export.json,也没有 meta/info.json —— 不是 LeRobotDataset 目录</span>
          )}
        </div>

        {manifest && !meta.ours ? (
          // Not a failure banner: this is the "partial dataset" state. The
          // operator should know that a green verdict here means "the parquet
          // reads back", not "the parquet matches the source".
          <div className="body">
            <Banner tone="warn">
              这是 lerobot 的数据集,但没有 so-snake 的 <span className="mono">export.json</span>
              。回放仍然可用(parquet 是数据来源);校验会跑 round-trip 和时间轴,但
              <b> 不会</b>跟原始 take 比对 —— 因为没有 source mapping。要做完整校验,
              在「录制」页用覆盖导出(<span className="mono">覆盖同名数据集</span>)
              重新写一次。
            </Banner>
          </div>
        ) : null}

        <div className="body">
          <VerdictPanel verdict={verdict} progress={verifyProgress} running={runningVerify} />
        </div>
      </section>

      {/* Replay card: shown whenever a parquet exists. The manifest is the
          one thing that maps a dataset episode to a source take, and replay
          does not need that mapping -- the parquet has fps and action space
          on its own. Hiding replay for foreign datasets was the wrong
          default: it forced the operator to drop the parquet and start over
          just to run a check. */}
      {manifest ? (
        <Card
          title="在机械臂上回放"
          actions={
            replaying ? <Pill tone="ok" live>正在回放</Pill> : <span />
          }
        >
          <Banner tone="info">
            回放走的是同一套安全层(归位、限速、关节/工作区限位、MuJoCo 网格干涉检查)——
            一次回放既能确认导出是对的,也能确认策略确实能驱动这台臂。
            {!meta.ours ? (
              <>
                {" "}
                没有 manifest 时回放照样能跑 —— 用的 fps 和 action space 来自 lerobot 的
                <span className="mono"> meta/info.json</span>,源 take 名称会留空。
              </>
            ) : null}
          </Banner>

          <Field
            label="回放第几条"
            hint={
              meta.ours
                ? "dataset 内的索引 — manifest 里第 i 个 entry 对应第 i 条原始 take"
                : "dataset 内的索引 — 没有 manifest,这里只有 lerobot 自己记录的顺序"
            }
          >
            {episodeCount > 1 ? (
              <select
                value={episodeIndex}
                disabled={replaying}
                onChange={(event) => setEpisodeIndex(Number(event.target.value))}
              >
                {Array.from({ length: episodeCount }, (_, i) => (
                  <option key={i} value={i}>
                    {i}
                    {meta.ours
                      ? ` — ${manifest.episode_ids[i] ?? "(no source)"}`
                      : ""}
                  </option>
                ))}
              </select>
            ) : (
              <input value="0" disabled />
            )}
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
              disabled={replaying}
              onChange={(event) => setSpeed(Number(event.target.value))}
            />
          </Field>

          <RigControls
            rig={rig}
            onChange={setRig}
            config={config}
            disabled={!ready}
            clampDisabled={!ready}
            showSource={false}
          />
          {blocked && ready ? <Banner tone="warn">{blocked}</Banner> : null}
          {held ? (
            <Banner tone="info">
              机械臂已归位并<strong>保持力矩</strong>,可以直接回放;回放结束后仍然保持力矩,要松开就点「停止 / 卸力」。
            </Banner>
          ) : null}
          {!validEpisode ? (
            <Banner tone="warn">这个数据集没有第 {episodeIndex} 条。</Banner>
          ) : null}

          <div className="row">
            <button
              className="btn primary"
              disabled={!ready || !validEpisode || Boolean(blocked) || replaying || runningVerify}
              onClick={() => void replay()}
            >
              {replaying ? "回放中…" : "开始回放"}
            </button>
            <button className="btn danger" disabled={idle} onClick={() => void run(api.stop)}>
              {held ? "停止 / 卸力" : "停止"}
            </button>
          </div>

          {/* When a replay is running, the snapshot's `replay` block has the
              live phase. The recordings page shows it; here we just say so,
              and the snapshot's replay card on the status bar covers the
              detail. The detail comes back on the recordings page after the
              replay finishes. */}
          {replaying ? (
            <div className="hint dim small" style={{ marginTop: 8 }}>
              进度看顶部状态栏;结束后结果会出现在「录制」页。
            </div>
          ) : null}
        </Card>
      ) : null}

      {/* "用了哪些原始 take"折叠: only when we have a manifest we wrote.
          A foreign dataset has nothing to fold -- showing it would be a
          claim we cannot back. */}
      {meta.ours && manifest ? (
        <details className="card fold">
          <summary>用了哪些原始 take</summary>
          <div className="body">
            <div className="dim small" style={{ marginBottom: 8 }}>
              manifest 记录了 dataset episode i 来自哪一条原始 take ——
              这条映射是「导出」和「回放」之间的唯一桥梁。
            </div>
            <ul className="tight">
              {manifest.episode_ids.map((sourceId, i) => (
                <li key={i} className="mono small">
                  <span className="dim">[{i}]</span> {sourceId || "(missing)"}
                </li>
              ))}
            </ul>
          </div>
        </details>
      ) : null}
    </>
  );
}

/** The verdict, in four states: running, never, ok, partial.
 *
 *  * "校验中" polls until the job leaves the verifying phase; the report is
 *    still on the previous run's text and is not shown until that phase ends,
 *    so the operator does not see a half-written verdict.
 *  * "未校验" is a state, not a missing button: a freshly exported dataset is
 *    verified by default, but one that predates this page was not.
 *  * "校验过 / 部分" is the missing-input case: round-trip + time axis pass and
 *    source-fidelity is recorded in `skipped`, either because there is no
 *    manifest to map episodes to takes or because a take has since been deleted
 *    from the store. Treated as `warn`, not `ok`, because "the parquet reads
 *    back" is not the same claim as "the parquet matches the source" -- and not
 *    as `error`, because neither absence changes a byte of the dataset.
 *  * "校验过 / 失败" shows the cached answer plus the staleness flag, because
 *    a dataset re-exported in place would otherwise stay green over changed
 *    bytes. */
function VerdictPanel({
  verdict,
  progress,
  running
}: {
  verdict: DatasetMeta["verdict"];
  progress: ExportProgress | null;
  running: boolean;
}) {
  if (running && progress) {
    return (
      <>
        <div className="row">
          <progress value={progress.episodes_done} max={Math.max(progress.episodes_total, 1)} />
          <span className="mono">
            {progress.episodes_done}/{progress.episodes_total} 集 · {progress.frames_done} 帧
          </span>
        </div>
        <div className="hint dim small">{progress.current_episode}</div>
      </>
    );
  }

  if (!verdict) {
    return (
      <Banner tone="info">
        这个数据集还没校验过。校验会读回每行 parquet 和每路视频,核对能不能还原成当时记录的指令 —— 一次几十秒到几分钟,取决于大小。
      </Banner>
    );
  }

  const skipped = verdict.skipped ?? [];
  const isPartial = verdict.ok && skipped.length > 0;
  const tone = verdict.stale ? "warn" : isPartial ? "warn" : verdict.ok ? "ok" : "error";
  const headline = verdict.stale
    ? "校验已过期"
    : isPartial
      ? `部分校验 (round-trip 通过,${skipped.length} 项跳过)`
      : verdict.ok
        ? "校验通过"
        : `校验未通过 (${verdict.issues.length} 项)`;

  return (
    <>
      <Banner tone={tone}>{headline}</Banner>
      {verdict.stale ? (
        <Banner tone="warn">
          这份结果校验于 <span className="mono">{shortTime(verdict.verified_at * 1000)}</span>,
          数据集在 <span className="mono">{shortTime(verdict.dataset_mtime * 1000)}</span> 又被写过 ——
          缓存的校验结果不再可信,重新校验一次。
        </Banner>
      ) : null}
      {isPartial ? (
        <Banner tone="warn">
          round-trip / 时间轴 / 视频帧数 都过了 —— 但下面这些项没跑,因为前置数据不在:
          <ul className="tight">
            {skipped.map((note, i) => (
              <li key={i}>{note}</li>
            ))}
          </ul>
          源 take 还在的话,重新导出(覆盖同名数据集)会补上 source mapping,下一次校验就是完整的;
          take 已经删了的话,这份数据集就只能这样了 —— 照样能训,只是再也没法和当时的录制对账。
        </Banner>
      ) : null}
      <div className="stats">
        {/* Coverage first, because it qualifies every number after it: the误差
            columns are maxima over the episodes that were compared, and "0.00e+0
            over 0 of 1 集" is not the same claim as "0.00e+0 over 1 of 1". */}
        <Stat
          label="源比对"
          value={`${verdict.episodes_compared}/${verdict.n_episodes}`}
          unit="集"
          small
        />
        <Stat label="状态误差" value={fmt(verdict.state_max_abs_error)} small />
        <Stat label="动作误差" value={fmt(verdict.action_max_abs_error)} small />
        <Stat
          label="target 位置误差 max"
          value={fmt(verdict.target_position_error_max_m * 1000)}
          unit="mm"
          small
        />
        <Stat
          label="target 角度误差 max"
          value={fmt(npDeg(verdict.target_angle_error_max_rad))}
          unit="°"
          small
        />
        <Stat
          label="夹爪误差 max"
          value={fmt(verdict.gripper_error_max_deg)}
          unit="°"
          small
        />
        <Stat
          label="时间轴误差 max"
          value={fmt(verdict.timestamp_max_error_s * 1e6)}
          unit="µs"
          small
        />
        <Stat
          label="校验时间"
          value={shortTime(verdict.verified_at * 1000)}
          small
        />
      </div>
      {!verdict.ok ? (
        <Banner tone="error">
          不能直接拿去训:
          <ul className="tight">
            {verdict.issues.map((issue, i) => (
              <li key={i}>{issue}</li>
            ))}
          </ul>
        </Banner>
      ) : null}
      {Object.keys(verdict.video_frames).length > 0 ? (
        <div className="hint mono small">
          视频帧数:{Object.entries(verdict.video_frames).map(([role, n]) => `${role} ${n}`).join(" · ")}
        </div>
      ) : null}
    </>
  );
}

function verdictLabel(dataset: DatasetMeta): string {
  if (!dataset.verdict) return "未校验";
  if (dataset.verdict.stale) return "过期";
  const skipped = dataset.verdict.skipped ?? [];
  if (dataset.verdict.ok && skipped.length > 0) return "部分校验";
  return dataset.verdict.ok ? "可回放" : "不可用";
}

function verdictBadge(dataset: DatasetMeta): "ok" | "warn" | "danger" | "neutral" {
  if (!dataset.verdict) return "neutral";
  if (dataset.verdict.stale) return "warn";
  const skipped = dataset.verdict.skipped ?? [];
  if (dataset.verdict.ok && skipped.length > 0) return "warn";
  return dataset.verdict.ok ? "ok" : "danger";
}

/** mtime is seconds-since-epoch on the wire, Date wants milliseconds. The
 *  caller hands us either; this normalises. */
/** A wall-clock stamp for the operator: `HH:MM` today, `MM-DD` before that.
 *
 * Local time, not UTC. Every number this formats is compared against a clock
 * on the wall of the room the arm is in -- "the dataset changed at 14:36" is
 * only useful if 14:36 is when the operator was standing there. `toISOString`
 * was doing the formatting before, so a take exported at 22:36 read as 14:36,
 * and "today" flipped eight hours early. Takes milliseconds, like `Date`. */
function shortTime(value: number | string): string {
  const ms = typeof value === "string" ? Date.parse(value) : value;
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

function fmt(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toExponential(2);
}

function npDeg(value: number | undefined): number {
  return value === undefined || !Number.isFinite(value) ? NaN : (value * 180) / Math.PI;
}
