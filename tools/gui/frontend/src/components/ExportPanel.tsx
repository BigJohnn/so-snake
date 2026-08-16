import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type ExportBody } from "../api";
import { Banner, Field, Pill, Segmented } from "./ui";
import type { ActionSpace, ExportPlan, ExportProgress, TaskSummary } from "../types";

/* Exporting a training set, from the page where takes are judged.
 *
 * It belongs here and not on its own page because it is the last step of the
 * same loop: the operator watches takes, deletes the bad ones, labels the rest,
 * and then turns what survived into a dataset. Putting the button anywhere else
 * would mean leaving the page where the decision about *which* takes was made.
 *
 * Three things this panel insists on, all of them learned from the data:
 *
 *   * **one task per dataset.** The store holds several skills. A policy
 *     trained across two of them learns their average, so the selection is a
 *     task label and there is no "export everything" button.
 *   * **dry run before write.** Screening is a second of arithmetic; the export
 *     is minutes of video transcode. Finding out there that a take lost its
 *     video alignment is the cheapest place to find out.
 *   * **the export is not finished until it has been read back.** Every failure
 *     that makes a dataset unusable -- an unwritten parquet footer, a video a
 *     frame short, a time axis built from a rate nothing ran at -- looks like
 *     success at the moment of writing. So the job verifies by default and this
 *     panel reports that verdict, not the write.
 */

const POLL_MS = 1000;

export function ExportPanel() {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [datasetRoot, setDatasetRoot] = useState("");
  const [task, setTask] = useState("");
  const [repoId, setRepoId] = useState("");
  const [actionSpace, setActionSpace] = useState<ActionSpace>("delta");
  const [plan, setPlan] = useState<ExportPlan | null>(null);
  const [planning, setPlanning] = useState(false);
  const [progress, setProgress] = useState<ExportProgress | null>(null);
  const [error, setError] = useState("");
  const [showLog, setShowLog] = useState(false);
  const logRef = useRef<HTMLPreElement | null>(null);

  const loadTasks = useCallback(async () => {
    try {
      const payload = await api.exportTasks();
      setTasks(payload.tasks);
      setDatasetRoot(payload.dataset_root);
      setTask((current) => current || payload.tasks[0]?.task || "");
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void loadTasks();
  }, [loadTasks]);

  // Poll only while something is running. An export takes minutes, and the job
  // outlives the request that started it, so the first poll on mount is also
  // what reattaches this panel to an export already in flight.
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

  useEffect(() => {
    if (showLog && logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [showLog, progress?.log?.length]);

  const selected = tasks.find((t) => t.task === task);
  // Suggested, not imposed: a repo id is a name the operator will type into a
  // training command later, and a slug derived from a Chinese task label is not
  // one they would have chosen.
  const effectiveRepoId = repoId.trim() || `so_snake/${slugify(task)}`;

  const body = (): ExportBody => ({
    repo_id: effectiveRepoId,
    task: task || null,
    action_space: actionSpace,
    verify: true
  });

  const dryRun = async () => {
    setPlanning(true);
    setError("");
    setPlan(null);
    try {
      setPlan(await api.exportPlan(body()));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    } finally {
      setPlanning(false);
    }
  };

  const start = async () => {
    setError("");
    try {
      setProgress(await api.startExport(body()));
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  };

  const cancel = async () => {
    try {
      setProgress(await api.cancelExport());
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
    }
  };

  const running = Boolean(progress?.running);
  const report = progress?.report;
  const verified = progress?.verify_report;

  return (
    <section className="card">
      <header>
        <h2>导出训练集</h2>
        <div className="spacer">
          {progress && progress.phase !== "idle" ? (
            <Pill tone={phaseTone(progress.phase)} live={running}>
              {PHASE_LABELS[progress.phase] ?? progress.phase}
            </Pill>
          ) : null}
        </div>
      </header>
      <div className="body">
        {error ? <Banner tone="error">{error}</Banner> : null}

        <Field
          label="技能 (task)"
          hint="一次导一个技能 —— 一个库里混着几种任务,合起来训出的策略学的是它们的平均"
        >
          <select
            value={task}
            disabled={running}
            onChange={(event) => {
              setTask(event.target.value);
              setPlan(null);
            }}
          >
            {tasks.length === 0 ? <option value="">(还没有录制)</option> : null}
            {tasks.map((t) => (
              <option key={t.task} value={t.task}>
                {t.task || "(未标注)"} — {t.takes} 条 / {Math.round(t.seconds)} 秒
              </option>
            ))}
          </select>
        </Field>

        {selected && selected.takes < 20 ? (
          <Banner tone="warn">
            只有 {selected.takes} 条。够跑通流程,不够训出稳定策略 —— 一般要 50 条上下,
            而且物体位置要有变化。
          </Banner>
        ) : null}

        <Field label="动作空间" hint="delta 锚在实测位姿上,rollout 会自纠;absolute 是对照组">
          <Segmented<ActionSpace>
            value={actionSpace}
            disabled={running}
            onChange={(value) => {
              setActionSpace(value);
              setPlan(null);
            }}
            options={[
              { value: "delta", label: "delta 增量" },
              { value: "absolute", label: "absolute 绝对" }
            ]}
          />
        </Field>

        <Field label="repo id" hint={`留空就用 ${`so_snake/${slugify(task)}`};写到 ${datasetRoot}`}>
          <input
            value={repoId}
            disabled={running}
            placeholder={`so_snake/${slugify(task)}`}
            onChange={(event) => setRepoId(event.target.value)}
          />
        </Field>

        <div className="row">
          <button className="btn" disabled={running || planning || !task} onClick={() => void dryRun()}>
            {planning ? "试算中…" : "试算(不写盘)"}
          </button>
          <button
            className="btn primary"
            disabled={running || !task}
            onClick={() => void start()}
          >
            导出
          </button>
          {running ? (
            <button className="btn small" onClick={() => void cancel()}>
              取消
            </button>
          ) : null}
        </div>

        {plan ? <PlanSummary plan={plan} /> : null}

        {progress && progress.phase !== "idle" ? (
          <div className="export-progress">
            {running ? (
              <>
                <div className="row">
                  <progress
                    value={progress.episodes_done}
                    max={Math.max(progress.episodes_total, 1)}
                  />
                  <span className="mono">
                    {progress.episodes_done}/{progress.episodes_total} 条 ·{" "}
                    {progress.frames_done} 帧
                  </span>
                </div>
                <div className="hint">
                  {progress.phase === "verifying"
                    ? "正在读回校验 —— 写完不等于能用"
                    : progress.current_episode}
                </div>
              </>
            ) : null}

            {progress.phase === "failed" && progress.error ? (
              <Banner tone="error">{progress.error}</Banner>
            ) : null}
            {progress.phase === "cancelled" ? (
              <Banner tone="warn">
                已取消。写出去的 {report?.n_episodes ?? 0} 条是完整的,能直接加载。
              </Banner>
            ) : null}

            {report && !running ? (
              <ReportSummary report={report} verified={verified ?? null} />
            ) : null}

            {progress.log.length ? (
              <>
                <button className="btn small" onClick={() => setShowLog(!showLog)}>
                  {showLog ? "收起日志" : `日志 (${progress.log.length})`}
                </button>
                {showLog ? (
                  <pre className="export-log" ref={logRef}>
                    {progress.log.join("\n")}
                  </pre>
                ) : null}
              </>
            ) : null}
          </div>
        ) : null}
      </div>
    </section>
  );
}

/** The dry run's answer: what would go in, what would not, and at what rate. */
function PlanSummary({ plan }: { plan: ExportPlan }) {
  const { report } = plan;
  const included = report.episodes.filter((e) => e.included);
  // Takes that did not hold the rate they were configured for. This is the only
  // thing that explains a dataset rate lower than expected, and it is a property
  // of the recording -- so the message has to say "re-record", not "try again".
  const behind = included.filter(
    (e) => e.configured_hz > 0 && e.measured_fps < e.configured_hz * 0.95
  );
  const configured = [...new Set(behind.map((e) => Math.round(e.configured_hz)))];
  return (
    <div className="export-plan">
      <Banner tone={plan.usable > 0 ? "info" : "error"}>
        {plan.usable > 0 ? (
          <>
            可导出 <b>{report.n_episodes}</b> 条 / <b>{report.n_frames}</b> 帧,
            帧率 <b>{report.fps} Hz</b>(实测出来的,不是配置值)。
          </>
        ) : (
          <>没有一条能过筛 —— 下面是原因。</>
        )}
      </Banner>
      {behind.length ? (
        <Banner tone="warn">
          {behind.length}/{included.length} 条录制时配的是 {configured.join("、")} Hz 但没跑住。
          这已经烙在那些 take 里了 —— 导出报的是臂当时真实的速度,<b>重导不会变高</b>,
          只能重录。
        </Banner>
      ) : null}
      {report.skipped.length ? (
        <details className="export-skipped">
          <summary>{report.skipped.length} 条被筛掉</summary>
          <ul className="tight">
            {report.skipped.map((entry) => (
              <li key={entry.episode_id}>
                <span className="mono">{entry.episode_id}</span> — {entry.reason}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {report.action_stats?.gripper?.transitions === 0 ? (
        <Banner tone="warn">
          夹爪自始至终没开合过 —— 这批数据里没有抓取动作,训不出抓取。
        </Banner>
      ) : null}
    </div>
  );
}

/** What was written, and -- the part that matters -- whether it reads back. */
function ReportSummary({
  report,
  verified
}: {
  report: NonNullable<ExportProgress["report"]>;
  verified: ExportProgress["verify_report"];
}) {
  return (
    <div className="export-report">
      <div className="hint mono">{report.dataset_path}</div>
      <div className="row">
        <Pill tone="accent">
          {report.n_episodes} 条 / {report.n_frames} 帧 @ {report.fps} Hz
        </Pill>
        <Pill tone="neutral">{report.action_space}</Pill>
      </div>

      {verified ? (
        verified.ok ? (
          <Banner tone="ok">
            已读回校验:导出的行能还原成录制时的指令(位置误差 &lt;{" "}
            {(verified.target_position_error_max_m * 1e6).toPrecision(2)} µm),
            每路视频帧数与行数一致。这份数据可回放。
          </Banner>
        ) : (
          <Banner tone="error">
            校验没过 —— 这份数据不能直接拿去训:
            <ul className="tight">
              {verified.issues.map((issue) => (
                <li key={issue}>{issue}</li>
              ))}
            </ul>
          </Banner>
        )
      ) : null}

      {verified?.ok ? (
        <div className="hint">
          回放验证:
          <span className="mono">
            {" "}
            scripts/replay_lerobot_dataset.py --dataset {report.dataset_path}
          </span>
        </div>
      ) : null}
    </div>
  );
}

const PHASE_LABELS: Record<string, string> = {
  exporting: "导出中",
  verifying: "校验中",
  done: "完成",
  failed: "失败",
  cancelled: "已取消"
};

function phaseTone(phase: string) {
  if (phase === "failed") return "danger" as const;
  if (phase === "cancelled") return "warn" as const;
  if (phase === "done") return "ok" as const;
  return "accent" as const;
}

/** A repo id has to survive being a directory name and a CLI argument.
 *
 * The task labels on this bench are Chinese, which slugifies to nothing useful,
 * so a label with no ASCII in it falls back to a fixed name rather than an
 * empty string -- the operator can always type their own. */
function slugify(task: string): string {
  const slug = task
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return slug || "dataset";
}
