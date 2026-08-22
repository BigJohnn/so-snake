import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "../api";
import { Banner, Card, Empty, Field, Segmented, Stat } from "../components/ui";
import type { ActionSpace, AppConfig, BackendKind, ModelMeta, Snapshot, TrainingStatus } from "../types";

/** The rollout page puts identity and scene inputs before the start button:
 * executing a model is an experiment, so the operator must be able to answer
 * exactly which checkpoint, task, camera roles and safety budget were used. */
export function RolloutsPage({ config, snapshot }: { config: AppConfig; snapshot: Snapshot | null }) {
  const [models, setModels] = useState<ModelMeta[]>([]); const [checkpoint, setCheckpoint] = useState("");
  const [task, setTask] = useState(""); const [backend, setBackend] = useState<BackendKind>("mock");
  const [device, setDevice] = useState("mps"); const [steps, setSteps] = useState(300); const [actionSpace, setActionSpace] = useState<ActionSpace>("delta");
  const [third, setThird] = useState(""); const [wrist, setWrist] = useState(""); const [confirm, setConfirm] = useState(false);
  const [status, setStatus] = useState<TrainingStatus | null>(null); const [error, setError] = useState("");
  const reload = useCallback(async () => { try { const [m, s] = await Promise.all([api.models(), api.trainingStatus()]); const ready = m.models.filter((x) => x.ready); setModels(ready); setCheckpoint((v) => ready.some((x) => x.checkpoint === v) ? v : ready[0]?.checkpoint ?? ""); setStatus(s); } catch (e) { setError(e instanceof ApiError ? e.message : String(e)); } }, []);
  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => { const timer = window.setInterval(() => void api.trainingStatus().then(setStatus).catch(() => undefined), 1000); return () => window.clearInterval(timer); }, []);
  const running = Boolean(status?.running && status.kind === "rollout");
  const selectedModel = models.find((model) => model.checkpoint === checkpoint);
  const localUnsupported = selectedModel?.policy === "pi05" && device.trim().toLowerCase() === "mps";
  const start = async () => { try { const cameras: Record<string, string> = {}; if (third) cameras.third_person = third; if (wrist) cameras.wrist = wrist; setStatus(await api.startPolicyRollout({ checkpoint, task, action_space: actionSpace, backend, cameras, steps, device, confirm_real: confirm })); setError(""); } catch (e) { setError(e instanceof ApiError ? e.message : String(e)); } };
  const armBusy = snapshot?.mode !== "idle";
  return <div className="grid cols-2">
    {error ? <Banner tone="error">{error}</Banner> : null}
    <Card title="策略执行">
      <Field label="Checkpoint"><select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)} disabled={running}>{models.map((x) => <option key={x.checkpoint} value={x.checkpoint}>{x.name}</option>)}</select></Field>
      {selectedModel && Object.keys(selectedModel.roi ?? {}).length ? <Banner tone="info">已加载该模型训练集的 ROI；每路相机帧会在进入策略前按相同区域裁剪。</Banner> : null}
      <Field label="任务 / language prompt" hint="pi0.5 将此文字作为策略输入；ACT 也记录它，方便复现实验。"><input value={task} onChange={(e) => setTask(e.target.value)} disabled={running} placeholder="例如：抓起红色方块并放入盒中" /></Field>
      <Field label="动作契约"><Segmented value={actionSpace} onChange={setActionSpace} disabled={running} options={[{ value: "delta", label: "delta（默认）" }, { value: "absolute", label: "absolute" }]} /></Field>
      <div className="grid cols-2"><Field label="后端"><select value={backend} onChange={(e) => setBackend(e.target.value as BackendKind)} disabled={running}>{(["mock", "mujoco", "real"] as BackendKind[]).map((x) => <option key={x} value={x} disabled={!config.availability.backends[x].available}>{x}</option>)}</select></Field><Field label="推理设备"><input value={device} onChange={(e) => setDevice(e.target.value)} disabled={running} /></Field><Field label="最大步数"><input type="number" min="1" value={steps} onChange={(e) => setSteps(Number(e.target.value))} disabled={running} /></Field></div>
      <div className="grid cols-2"><Field label="第三人称相机"><input value={third} onChange={(e) => setThird(e.target.value)} placeholder="例如 0" disabled={running} /></Field><Field label="腕部相机"><input value={wrist} onChange={(e) => setWrist(e.target.value)} placeholder="例如 1" disabled={running} /></Field></div>
      {backend === "real" ? <label className="row small"><input type="checkbox" checked={confirm} onChange={(e) => setConfirm(e.target.checked)} /> 我确认工作区已清空，手在断电/急停旁；策略会让真机运动。</label> : null}
      <div className="row"><button className="btn primary" disabled={!checkpoint || !task || !third && !wrist || armBusy || running || localUnsupported || (backend === "real" && !confirm)} onClick={() => void start()}>开始 Rollout</button>{running ? <button className="btn danger" onClick={() => void api.stopTraining().then(setStatus)}>停止</button> : null}</div>
      {armBusy ? <Banner tone="warn">机械臂当前在 {snapshot?.mode}；先停止该会话才可执行策略。</Banner> : null}
      {localUnsupported ? <Banner tone="warn">pi0.5 不建议在本机 MPS 做实时策略执行；请用 CUDA 推理服务，或先用 ACT 验证机械臂和相机闭环。</Banner> : null}
    </Card>
    <Card title="Rollout 状态">{status?.kind === "rollout" ? <><div className="stats"><Stat label="模型" value={status.name} small /><Stat label="状态" value={status.phase} small /><Stat label="PID" value={status.pid || "—"} small /></div>{status.error ? <Banner tone="error">{status.error}</Banner> : null}<pre className="export-log">{status.log.join("\n") || "加载 checkpoint…"}</pre></> : <Empty>选择一个完成训练的模型，配置相机后执行。每一步均通过任务空间、IK、关节速率与仿真网格间隙限制。</Empty>}</Card>
    <DeploymentAdvice policy={selectedModel?.policy || ""} />
  </div>;
}

function DeploymentAdvice({ policy }: { policy: string }) {
  const pi = policy === "pi05";
  return <Card title="推理部署建议">
    <div className="stats"><Stat label="模型" value={pi ? "pi0.5" : policy === "act" ? "ACT" : "选择 checkpoint 后判定"} small /><Stat label="本机 M1" value={pi ? "不建议" : "可用"} small /><Stat label="推荐架构" value={pi ? "边缘控制 + CUDA 推理" : "本机或 Orin"} small /></div>
    {pi ? <div className="deployment-advice">
      <p><strong>最低可试：</strong>CUDA GPU 12GB；只适合离线/单路验证，不能把它视作稳定的真机实时预算。</p>
      <p><strong>建议：</strong>RTX 4090 24GB 做推理；若需同时训练或更高并发，选 A100 40GB+。</p>
      <p><strong>边缘端：</strong>Jetson Orin AGX 64GB 可作为本地推理试验目标，但上线更建议 Orin 只负责相机、机械臂安全与缓存动作块，推理服务放同网段的 RTX 4090/A100；先实测端到端延迟再接真机。</p>
    </div> : <div className="deployment-advice"><p><strong>本机：</strong>ACT 可在 M1/MPS 上运行，适合台架验证。</p><p><strong>边缘端：</strong>Jetson Orin NX 16GB 是最低实用起点；Orin AGX 32/64GB 更适合双相机和留出安全余量。部署前仍应测量相机→推理→命令的端到端时延。</p></div>}
  </Card>;
}
