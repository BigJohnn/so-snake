import { useState } from "react";
import { api, previewUrl } from "../api";
import { useSeries, useTicker } from "../hooks";
import { SeriesPlot } from "../components/SeriesPlot";
import { Banner, Card, Empty, Field, JointBar, Pill, Stat } from "../components/ui";
import { DEFAULT_RIG, RigControls, rigBody, rigReady, type RigState } from "../components/RigControls";
import type { AppConfig, Snapshot, StartPoseStatus, Telemetry } from "../types";

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
  // 600 frames at 30 Hz is 20 s, which is a demonstration rather than a clip.
  // Fixed by default because takes an operator ends by hand vary by seconds and
  // that variance ends up in the dataset; 0 restores "until I say stop".
  const [takeSteps, setTakeSteps] = useState(600);
  const [takeCount, setTakeCount] = useState(10);

  const mode = snapshot?.mode ?? "idle";
  const idle = mode === "idle";
  const teleop = mode === "teleop";
  // Homed and energized. Not idle -- the arm is standing there under torque --
  // but a session may start straight from it, which is the point: torque is
  // never dropped between homing and teleoperating.
  const held = mode === "held";
  const ready = idle || held;
  const recording = Boolean(snapshot?.recording.recording);
  const takes = snapshot?.takes;
  const lastTake = snapshot?.last_take;
  const blocked = rigReady(rig, config);
  const hz = config.teleop.control_hz;

  const series = useSeries(mode !== "idle");
  // Every role that something can fill: a real camera opened for it, or the
  // simulator's camera of the same name. The real one wins on the server side,
  // so this only decides which panes are worth asking for.
  const liveCameras = snapshot?.cameras?.connected ?? [];
  const simCameras = (snapshot?.spec?.backend ?? "") === "mujoco" ? config.cameras : [];
  const panes = config.cameras.filter(
    (role) => liveCameras.includes(role) || simCameras.includes(role)
  );
  // Kept on through the automatic homing between takes: the cameras are still
  // open, and watching the arm walk back is exactly when the operator wants to
  // see them -- that is when they check the scene is reset for the next take.
  const previewOn = (teleop || mode === "homing") && panes.length > 0;
  // 8 fps. The gateway throttles to 10 and serves the cache in between, so
  // asking faster would only spend CPU that the video encoder wants while a
  // take is recording.
  const tick = useTicker(120, previewOn);

  const latest = (snapshot?.latest ?? {}) as Partial<Telemetry>;
  const hasTelemetry = typeof latest.index === "number";

  return (
    <div className="grid cols-2" style={{ gridTemplateColumns: "360px minmax(0, 1fr)" }}>
      {/* -------------------------------------------------------- left column */}
      <div className="grid" style={{ alignContent: "start" }}>
        <Card title="会话">
          {/* Locked while the arm is held as well as while it is running: the
              held backend is a specific arm on a specific port, and changing
              which one is being asked for would only earn a refusal. */}
          <RigControls
            rig={rig}
            onChange={setRig}
            config={config}
            disabled={!idle}
            clampDisabled={!ready}
          />
          {blocked && ready ? <Banner tone="warn">{blocked}</Banner> : null}
          {held ? (
            <Banner tone="info">
              已归位并<strong>保持力矩</strong> —— 可直接启动遥操作(从 home 位姿开始),
              或点「停止 / 卸力」让机械臂松开。
            </Banner>
          ) : null}
          <div className="row">
            <button
              className="btn primary"
              disabled={!ready || Boolean(blocked)}
              onClick={() => void run(() => api.startSession(rigBody(rig)))}
            >
              启动遥操作
            </button>
            <button
              className="btn"
              disabled={!ready || Boolean(blocked)}
              title="以限速、无 IK 的方式走到 home 位姿,到位后保持力矩"
              onClick={() => void run(() => api.home(rigBody(rig)))}
            >
              归位
            </button>
            <button className="btn danger" disabled={idle} onClick={() => void run(api.stop)}>
              {held ? "停止 / 卸力" : "停止"}
            </button>
          </div>

          <StartPoseField
            startPose={snapshot?.start_pose}
            // Needs a live arm to read: teleop (the useful case -- fly there and
            // press it) or held. Idle has nothing to read.
            canCapture={teleop || held}
            onCapture={() => void run(api.captureStartPose)}
          />
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
          <div className="grid cols-2">
            <Field
              label="每条帧数"
              hint={takeSteps > 0 ? `≈ ${(takeSteps / hz).toFixed(1)} s,录满自动保存` : "0 = 手动停止"}
            >
              <input
                type="number"
                min={0}
                max={100000}
                step={30}
                value={takeSteps}
                disabled={recording}
                onChange={(event) => setTakeSteps(Math.max(0, Number(event.target.value) || 0))}
              />
            </Field>
            <Field label="目标条数" hint="只用于计数提示,不会自动开录">
              <input
                type="number"
                min={0}
                max={1000}
                value={takeCount}
                disabled={recording}
                onChange={(event) => setTakeCount(Math.max(0, Number(event.target.value) || 0))}
              />
            </Field>
          </div>
          {lastTake?.pending ? (
            // The verdict on a take that ended on its own. Shown while the arm
            // walks home, which is exactly when the operator knows whether it
            // went well -- and before the next take overwrites their memory of
            // this one.
            <div className="banner info" style={{ display: "grid", gap: 8 }}>
              <div>
                刚录完 <span className="mono">{lastTake.id}</span> · {lastTake.n_steps} 帧 ·{" "}
                {lastTake.duration_s.toFixed(1)} s{lastTake.task ? ` · ${lastTake.task}` : ""}
                <br />
                这条要吗?(不选就默认保留)
              </div>
              <div className="row">
                <button className="btn primary" onClick={() => void run(() => api.decideLastTake(true))}>
                  保留
                </button>
                <button
                  className="btn danger"
                  title="从磁盘删掉这条 episode(含视频),并把它从本次计数里减掉"
                  onClick={() => void run(() => api.decideLastTake(false))}
                >
                  丢弃
                </button>
              </div>
            </div>
          ) : null}
          {takes && takes.done_count > 0 ? (
            <div className="row" style={{ marginBottom: 8 }}>
              <Pill tone={takes.target_count && takes.done_count >= takes.target_count ? "ok" : "accent"}>
                本次会话已录 {takes.done_count}
                {takes.target_count ? ` / ${takes.target_count}` : ""} 条
              </Pill>
            </div>
          ) : null}
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
                // Disabled during the automatic homing between takes (mode is
                // "homing" while the arm walks back), which is also the honest
                // signal that the next take is not ready to start yet.
                disabled={!teleop}
                onClick={() =>
                  void run(() =>
                    api.startRecording({ name, task, steps: takeSteps, target_count: takeCount })
                  )
                }
              >
                ● 开始录制
              </button>
            )}
          </div>
          {!teleop && !recording ? (
            <div className="hint dim small" style={{ marginTop: 8 }}>
              {mode === "homing"
                ? "正在自动归位 …… 到位后即可开始下一条。"
                : "先启动遥操作会话再录制。一次会话可以连续录多条。"}
            </div>
          ) : null}
          {teleop && !recording ? (
            <div className="hint dim small" style={{ marginTop: 8 }}>
              录满 {takeSteps || "—"} 帧自动保存并归位,然后等你再按「开始录制」。
            </div>
          ) : null}
          {recording ? (
            <div className="stats" style={{ marginTop: 12 }}>
              <Stat
                label="已录帧数"
                value={snapshot?.recording.steps ?? 0}
                unit={takeSteps ? `/ ${takeSteps}` : ""}
                small
              />
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
          <Card title="相机">
            {previewOn ? (
              // Both roles at once rather than one behind a selector: the
              // operator is judging the wrist against the scene, and a toggle
              // makes that a memory test. Each pane is its own throttled key
              // on the gateway, so two panes cost two cached renders a second,
              // not two renders per poll.
              <div className={`grid${panes.length > 1 ? " cols-2" : ""}`}>
                {panes.map((role) => (
                  <div className="preview" key={role}>
                    <img
                      src={previewUrl(role, 480, 360, tick)}
                      alt={`${role} 相机`}
                      loading="lazy"
                    />
                    <div className="tag">
                      {liveCameras.includes(role) ? `${role} · 实拍` : `${role} · 仿真`}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <Empty>
                {!teleop
                  ? "启动一个会话以查看画面"
                  : "没有可显示的相机:在左侧扫描并指派,或用 mujoco backend 看仿真相机"}
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

/** The homing target, and the button that re-records it from the arm.
 *
 * Shown next to the 归位 button rather than on a settings page because it is
 * the thing that button does: without seeing which pose is recorded, "归位" is
 * a move to somewhere the operator has to remember.
 */
function StartPoseField({
  startPose,
  canCapture,
  onCapture
}: {
  startPose?: StartPoseStatus;
  canCapture: boolean;
  onCapture: () => void;
}) {
  const recorded = startPose?.source === "file";
  const joints = Object.entries(startPose?.joints_deg ?? {});

  return (
    <Field
      label="归位点"
      hint={
        recorded
          ? "遥操作途中飞到想要的位置,按下就把当前关节角记成归位点(写 assets/so100_start_pose.json)"
          : "还没记录过,归位走配置里的 home_joints_deg"
      }
    >
      <div className="row">
        <button
          className="btn"
          disabled={!canCapture}
          title={
            canCapture
              ? "读当前关节角并写入 start pose;不发运动指令"
              : "需要一条在线的臂:先启动遥操作,或先归位"
          }
          onClick={onCapture}
        >
          记录当前位姿为归位点
        </button>
        <Pill tone={recorded ? "ok" : "neutral"}>{recorded ? "已记录" : "用配置默认值"}</Pill>
      </div>

      {startPose?.error ? <Banner tone="warn">{startPose.error}</Banner> : null}
      {recorded && startPose?.in_workspace_box === false ? (
        <Banner tone="info">
          这个归位点在遥操作工作区盒子之外 —— 可以用,但遥操作会从这里限幅飞回盒内。
        </Banner>
      ) : null}
      {joints.length > 0 ? (
        <div className="hint dim small mono" style={{ marginTop: 6 }}>
          {joints.map(([name, value]) => `${name} ${value.toFixed(1)}°`).join("  ")}
          {startPose?.recorded_at ? ` · ${startPose.recorded_at.slice(0, 16).replace("T", " ")}` : ""}
        </div>
      ) : null}
    </Field>
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
