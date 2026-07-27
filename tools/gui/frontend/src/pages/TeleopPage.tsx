import { useState } from "react";
import { api, previewUrl } from "../api";
import { useSeries, useTicker } from "../hooks";
import { SeriesPlot } from "../components/SeriesPlot";
import { Banner, Card, Empty, Field, JointBar, Pill, Stat } from "../components/ui";
import { DEFAULT_RIG, RigControls, rigBody, rigReady, type RigState } from "../components/RigControls";
import type { AppConfig, Snapshot, Telemetry } from "../types";

const AXES = ["x", "y", "z"] as const;

export function TeleopPage({
  config,
  snapshot,
  run
}: {
  config: AppConfig;
  snapshot: Snapshot | null;
  run: (action: () => Promise<unknown>) => Promise<void>;
}) {
  const [rig, setRig] = useState<RigState>(DEFAULT_RIG);
  const [name, setName] = useState("");
  const [task, setTask] = useState("");
  const [camera, setCamera] = useState("third_person");

  const mode = snapshot?.mode ?? "idle";
  const idle = mode === "idle";
  const teleop = mode === "teleop";
  const recording = Boolean(snapshot?.recording.recording);
  const blocked = rigReady(rig, config);

  const series = useSeries(mode !== "idle");
  const previewOn = teleop && (snapshot?.spec?.backend ?? "") === "mujoco";
  const tick = useTicker(120, previewOn);

  const latest = (snapshot?.latest ?? {}) as Partial<Telemetry>;
  const hasTelemetry = typeof latest.index === "number";

  return (
    <div className="grid cols-2" style={{ gridTemplateColumns: "360px minmax(0, 1fr)" }}>
      {/* -------------------------------------------------------- left column */}
      <div className="grid" style={{ alignContent: "start" }}>
        <Card title="会话">
          <RigControls rig={rig} onChange={setRig} config={config} disabled={!idle} />
          {blocked && idle ? <Banner tone="warn">{blocked}</Banner> : null}
          <div className="row">
            <button
              className="btn primary"
              disabled={!idle || Boolean(blocked)}
              onClick={() => void run(() => api.startSession(rigBody(rig)))}
            >
              启动遥操作
            </button>
            <button
              className="btn"
              disabled={!idle || Boolean(blocked)}
              title="以限速、无 IK 的方式走到 home 位姿"
              onClick={() => void run(() => api.home(rigBody(rig)))}
            >
              归位
            </button>
            <button className="btn danger" disabled={idle} onClick={() => void run(api.stop)}>
              停止
            </button>
          </div>
          {rig.source === "pro" && teleop ? (
            <Banner tone="info">按住 ZL 才会运动;松开则目标冻结,可以重新摆手。</Banner>
          ) : null}
        </Card>

        <Card title="录制">
          <Field label="名称">
            <input
              value={name}
              disabled={recording}
              placeholder="take 1"
              onChange={(event) => setName(event.target.value)}
            />
          </Field>
          <Field label="任务描述" hint="训练时的 language instruction,尽量写具体">
            <input
              value={task}
              disabled={recording}
              placeholder="把红色方块放进框里"
              onChange={(event) => setTask(event.target.value)}
            />
          </Field>
          <div className="row">
            {recording ? (
              <>
                <button
                  className="btn primary"
                  onClick={() => void run(() => api.stopRecording(true))}
                >
                  保存这条
                </button>
                <button
                  className="btn danger"
                  title="这条不要了 —— 直接丢弃,不写盘"
                  onClick={() => void run(() => api.stopRecording(false))}
                >
                  丢弃
                </button>
              </>
            ) : (
              <button
                className="btn record"
                disabled={!teleop}
                onClick={() => void run(() => api.startRecording({ name, task }))}
              >
                ● 开始录制
              </button>
            )}
          </div>
          {!teleop && !recording ? (
            <div className="hint dim small" style={{ marginTop: 8 }}>
              先启动遥操作会话再录制。一次会话可以连续录多条。
            </div>
          ) : null}
          {recording ? (
            <div className="stats" style={{ marginTop: 12 }}>
              <Stat label="已录帧数" value={snapshot?.recording.steps ?? 0} small />
              <Stat label="时长" value={(snapshot?.recording.duration_s ?? 0).toFixed(1)} unit="s" small />
            </div>
          ) : null}
        </Card>

        <Card title="日志" padded={false}>
          <div className="log body">
            {(snapshot?.events ?? []).length === 0 ? (
              <div className="dim">暂无</div>
            ) : (
              [...(snapshot?.events ?? [])].reverse().map((event, i) => (
                <div key={`${event.time}-${i}`}>
                  <span className="ts">{event.time}</span>
                  <span className={event.level}>{event.message}</span>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      {/* ------------------------------------------------------- right column */}
      <div className="grid" style={{ alignContent: "start" }}>
        <Card title="实时状态" padded={false}>
          {hasTelemetry ? (
            <>
              <div className="stats">
                <Stat
                  label="回路频率"
                  value={(latest.loop_hz ?? 0).toFixed(1)}
                  unit={`Hz / ${config.teleop.control_hz}`}
                />
                <Stat label="IK 位置误差" value={(latest.ik_position_error_mm ?? 0).toFixed(3)} unit="mm" />
                <Stat label="pitch 误差" value={(latest.ik_pitch_error_deg ?? 0).toFixed(2)} unit="°" />
                <Stat
                  label="离合 (ZL)"
                  value={<Pill tone={latest.clutch ? "ok" : "neutral"}>{latest.clutch ? "按下" : "松开"}</Pill>}
                  small
                />
                <Stat
                  label="网格离地"
                  value={latest.robot_mesh_min_z_m === null || latest.robot_mesh_min_z_m === undefined
                    ? "—"
                    : (latest.robot_mesh_min_z_m * 1000).toFixed(0)}
                  unit={latest.robot_mesh_min_z_m == null ? "" : "mm"}
                />
              </div>

              <div className="body">
                <div className="flags" style={{ marginBottom: 14 }}>
                  <Flag on={!latest.flags?.ik_converged} tone="warn" label="IK 未收敛" offLabel="IK 收敛" />
                  <Flag on={latest.flags?.workspace_clamped} tone="warn" label="工作区限幅" />
                  <Flag on={latest.flags?.atlas_pitch_clamped} tone="warn" label="atlas pitch 限幅" />
                  <Flag on={latest.flags?.atlas_roll_infeasible} tone="warn" label="roll 不可行" />
                  <Flag on={latest.flags?.joint_limit_clamped} tone="warn" label="关节限位" />
                  <Flag on={latest.flags?.joint_rate_clamped} tone="warn" label="关节限速" />
                  <Flag on={latest.flags?.command_safety_held} tone="danger" label="安全保持" />
                </div>
                {latest.safety_reason ? (
                  <Banner tone="warn">
                    指令被保持: <span className="mono">{latest.safety_reason}</span>
                  </Banner>
                ) : null}

                <div className="grid cols-2">
                  <div>
                    <div className="plot-title">
                      <span>任务位姿 (x, y, z)</span>
                      <span>m</span>
                    </div>
                    {AXES.map((axis, i) => (
                      <div className="joint" key={axis}>
                        <div className="name">{axis} 目标 / 实际</div>
                        <div className="track">
                          <div
                            className="fill"
                            style={{
                              left: 0,
                              width: `${axisPercent(latest.task_target?.[i], config.limits.pos_min_m[i], config.limits.pos_max_m[i])}%`
                            }}
                          />
                          <div
                            className="measured"
                            style={{
                              left: `${axisPercent(latest.achieved_task_pose?.[i], config.limits.pos_min_m[i], config.limits.pos_max_m[i])}%`
                            }}
                          />
                        </div>
                        <div className="val">{(latest.task_target?.[i] ?? 0).toFixed(3)}</div>
                      </div>
                    ))}
                    <div className="joint">
                      <div className="name">pitch</div>
                      <div className="track">
                        <div
                          className="measured"
                          style={{
                            left: `${axisPercent(
                              ((latest.task_target?.[3] ?? 0) * 180) / Math.PI,
                              config.limits.pitch_min_deg,
                              config.limits.pitch_max_deg
                            )}%`
                          }}
                        />
                      </div>
                      <div className="val">{(((latest.task_target?.[3] ?? 0) * 180) / Math.PI).toFixed(1)}°</div>
                    </div>
                    <div className="joint">
                      <div className="name">roll</div>
                      <div className="track">
                        <div
                          className="measured"
                          style={{ left: `${axisPercent(((latest.task_target?.[4] ?? 0) * 180) / Math.PI, -180, 180)}%` }}
                        />
                      </div>
                      <div className="val">{(((latest.task_target?.[4] ?? 0) * 180) / Math.PI).toFixed(1)}°</div>
                    </div>
                  </div>

                  <div>
                    <div className="plot-title">
                      <span>关节 —— 填充为指令,黄条为实测</span>
                      <span>deg</span>
                    </div>
                    {config.arm.joint_names.map((joint, i) => (
                      <JointBar
                        key={joint}
                        name={joint}
                        commanded={latest.commanded_joints_deg?.[i] ?? 0}
                        measured={latest.measured_joints_deg?.[i]}
                        limits={config.arm.joint_limits_deg[joint]}
                      />
                    ))}
                    <JointBar
                      name="gripper"
                      commanded={latest.gripper_cmd_deg ?? 0}
                      limits={config.arm.joint_limits_deg.gripper}
                    />
                  </div>
                </div>
              </div>
            </>
          ) : (
            <Empty>{idle ? "启动一个会话以查看实时状态" : "等待第一帧 ..."}</Empty>
          )}
        </Card>

        <div className="grid cols-2">
          <Card
            title="仿真相机"
            actions={
              <div className="row">
                {config.cameras.map((cam) => (
                  <button
                    key={cam}
                    className={`btn small${camera === cam ? " primary" : ""}`}
                    onClick={() => setCamera(cam)}
                  >
                    {cam}
                  </button>
                ))}
              </div>
            }
          >
            {previewOn ? (
              <div className="preview">
                <img src={previewUrl(camera, 640, 480, tick)} alt={`${camera} 相机`} />
                <div className="tag">{camera}</div>
              </div>
            ) : (
              <Empty>
                {snapshot?.spec?.backend === "real"
                  ? "真机相机采集尚未接入 (见「进度」页)"
                  : "仅 mujoco backend 有仿真相机"}
              </Empty>
            )}
          </Card>

          <Card title="趋势 (最近 20 s)">
            <SeriesPlot
              x={series.map((row) => row.t)}
              series={[{ label: "IK 位置误差", values: series.map((row) => row.pos_err_mm) }]}
              bands={[{ values: series.map((row) => row.clutch) }]}
              unit="mm · 蓝底为离合按下"
              height={100}
              yMin={0}
            />
            <SeriesPlot
              x={series.map((row) => row.t)}
              series={[{ label: "回路频率", values: series.map((row) => row.loop_hz) }]}
              reference={{ value: config.teleop.control_hz, label: `${config.teleop.control_hz} Hz` }}
              unit="Hz"
              height={100}
              yMin={0}
            />
            <SeriesPlot
              x={series.map((row) => row.t)}
              series={AXES.map((axis, i) => ({
                label: axis,
                values: series.map((row) => row.target[i])
              }))}
              unit="m"
              height={110}
            />
          </Card>
        </div>
      </div>
    </div>
  );
}

function Flag({
  on,
  tone,
  label,
  offLabel
}: {
  on?: boolean;
  tone: "warn" | "danger";
  label: string;
  offLabel?: string;
}) {
  if (!on) return offLabel ? <Pill tone="ok">{offLabel}</Pill> : null;
  return <Pill tone={tone}>{label}</Pill>;
}

function axisPercent(value: number | undefined, lo: number, hi: number): number {
  if (value === undefined || !Number.isFinite(value)) return 0;
  return (100 * (Math.min(Math.max(value, lo), hi) - lo)) / Math.max(hi - lo, 1e-9);
}
