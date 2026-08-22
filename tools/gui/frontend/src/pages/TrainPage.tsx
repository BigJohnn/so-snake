import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api";
import { Banner, Card, Empty, Field, Pill, Segmented, Stat } from "../components/ui";
import type { DatasetMeta, ModelMeta, PolicyKind, TrainingStatus } from "../types";

const POLL_MS = 1000;

/** Training is deliberately a small form: dataset → policy → compute budget →
 * named artefact.  Advanced CLI flags are not a useful UI surface until they
 * have a bench-proven default. */
export function TrainPage() {
  const [datasets, setDatasets] = useState<DatasetMeta[]>([]);
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [dataset, setDataset] = useState("");
  const [policy, setPolicy] = useState<PolicyKind>("act");
  const [name, setName] = useState("act_run");
  const [baseModel, setBaseModel] = useState("");
  const [piMode, setPiMode] = useState<"expert" | "full">("expert");
  const [autodl, setAutodl] = useState(false);
  const [host, setHost] = useState("");
  const [remoteUser, setRemoteUser] = useState("root");
  const [remotePort, setRemotePort] = useState("22");
  const [remoteRoot, setRemoteRoot] = useState("/root/autodl-tmp");
  const [remotePython, setRemotePython] = useState("python");
  const [wandb, setWandb] = useState(false);
  const [wandbProject, setWandbProject] = useState("so-snake");
  const [wandbEntity, setWandbEntity] = useState("");
  const [wandbNotes, setWandbNotes] = useState("");
  const [device, setDevice] = useState("mps");
  const [steps, setSteps] = useState(20000);
  const [batch, setBatch] = useState(8);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [error, setError] = useState("");

  const reload = useCallback(async () => {
    try {
      const [ds, ms, st] = await Promise.all([api.datasets(), api.models(), api.trainingStatus()]);
      setDatasets(ds.datasets); setModels(ms.models); setStatus(st);
      setDataset((value) => value && ds.datasets.some((x) => x.path === value) ? value : ds.datasets[0]?.path ?? "");
    } catch (cause) { setError(cause instanceof ApiError ? cause.message : String(cause)); }
  }, []);
  useEffect(() => { void reload(); }, [reload]);
  useEffect(() => {
    const timer = window.setInterval(() => void api.trainingStatus().then(setStatus).catch(() => undefined), POLL_MS);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => { if (status && !status.running) void reload(); }, [status?.running, status?.ended_at, reload]);

  const selectedDataset = useMemo(() => datasets.find((x) => x.path === dataset), [datasets, dataset]);
  const start = async () => {
    try { setStatus(await api.startTraining({
      dataset, policy, name, device: autodl ? "cuda" : device, steps, batch_size: batch, base_model: baseModel,
      wandb: { enabled: wandb, project: wandbProject, entity: wandbEntity, notes: wandbNotes },
      autodl: { enabled: autodl, host, user: remoteUser, port: remotePort, root: remoteRoot, python: remotePython }, pi_mode: piMode
    })); setError(""); }
    catch (cause) { setError(cause instanceof ApiError ? cause.message : String(cause)); }
  };
  const erase = async () => {
    const names = [...chosen];
    if (!names.length || !window.confirm(`永久删除 ${names.length} 个模型及其 checkpoints？`)) return;
    try { await api.deleteModels(names); setChosen(new Set()); await reload(); }
    catch (cause) { setError(cause instanceof ApiError ? cause.message : String(cause)); }
  };
  const toggle = (value: string) => setChosen((old) => { const next = new Set(old); next.has(value) ? next.delete(value) : next.add(value); return next; });

  return <div className="grid cols-2">
    {error ? <Banner tone="error">{error}</Banner> : null}
    <Card title="新建训练">
      <Field label="训练集" hint="训练集已通过导出与读回校验；一个 run 只选择一个数据集。">
        <select value={dataset} onChange={(e) => setDataset(e.target.value)} disabled={Boolean(status?.running)}>
          {datasets.map((x) => <option key={x.path} value={x.path}>{x.name} · {x.manifest?.n_episodes ?? "?"} 条</option>)}
        </select>
      </Field>
      {selectedDataset?.manifest?.roi && Object.keys(selectedDataset.manifest.roi).length ? <Banner tone="info">此训练集已应用 ROI：图像在导出时裁剪；同一裁剪参数会随完成的模型保存，并在 Rollout 自动应用。</Banner> : null}
      <Field label="Policy"><Segmented value={policy} onChange={(v) => { setPolicy(v); setName(v === "act" ? "act_run" : "pi05_run"); if (v === "pi05") setDevice("cuda"); }} disabled={Boolean(status?.running)} options={[{ value: "act", label: "ACT" }, { value: "pi05", label: "pi0.5" }]} /></Field>
      {policy === "pi05" ? <Field label="pi0.5 基础模型" hint="用于微调；可填写本地 checkpoint 路径或 Hugging Face repo ID。"><input value={baseModel} onChange={(e) => setBaseModel(e.target.value)} disabled={Boolean(status?.running)} placeholder="例如 lerobot/pi05_base" /></Field> : null}
      {policy === "pi05" ? <Field label="pi0.5 微调规模"><Segmented value={piMode} onChange={setPiMode} disabled={Boolean(status?.running)} options={[{ value: "expert", label: "4090 · expert-only" }, { value: "full", label: "A100 80GB · full" }]} /></Field> : null}
      <div className="grid cols-2">
        <Field label="输出名前缀" hint="启动时自动加时间戳，确保每个 run 和输出目录唯一。"><input value={name} onChange={(e) => setName(e.target.value)} disabled={Boolean(status?.running)} /></Field>
        <Field label="设备"><input value={device} onChange={(e) => setDevice(e.target.value)} disabled={Boolean(status?.running)} placeholder="mps / cuda / cpu" /></Field>
        <Field label="训练 steps"><input type="number" min="1" value={steps} onChange={(e) => setSteps(Number(e.target.value))} disabled={Boolean(status?.running)} /></Field>
        <Field label="batch size"><input type="number" min="1" value={batch} onChange={(e) => setBatch(Number(e.target.value))} disabled={Boolean(status?.running)} /></Field>
      </div>
      <label className="row small"><input type="checkbox" checked={autodl} onChange={(e) => setAutodl(e.target.checked)} disabled={Boolean(status?.running)} /> 在 AutoDL（SSH + rsync）训练</label>
      {autodl ? <div className="remote-profile">
        <Banner tone="info">使用已配置的 SSH key/agent；数据集上传、远程 CUDA 训练与 checkpoint 回传均由此 run 完成。W&amp;B 登录应已在远端完成。</Banner>
        <div className="grid cols-2">
          <Field label="AutoDL host"><input value={host} onChange={(e) => setHost(e.target.value)} placeholder="region-xx.autodl.com" /></Field>
          <Field label="SSH user"><input value={remoteUser} onChange={(e) => setRemoteUser(e.target.value)} /></Field>
          <Field label="SSH port"><input value={remotePort} onChange={(e) => setRemotePort(e.target.value)} /></Field>
          <Field label="远端工作目录"><input value={remoteRoot} onChange={(e) => setRemoteRoot(e.target.value)} /></Field>
          <Field label="远端 Python"><input value={remotePython} onChange={(e) => setRemotePython(e.target.value)} placeholder="/root/miniconda3/envs/lerobot/bin/python" /></Field>
        </div>
      </div> : null}
      <label className="row small"><input type="checkbox" checked={wandb} onChange={(e) => setWandb(e.target.checked)} disabled={Boolean(status?.running)} /> 记录到 Weights &amp; Biases</label>
      {wandb ? <div className="grid cols-2">
        <Field label="W&B project"><input value={wandbProject} onChange={(e) => setWandbProject(e.target.value)} /></Field>
        <Field label="W&B entity（可选）"><input value={wandbEntity} onChange={(e) => setWandbEntity(e.target.value)} /></Field>
        <Field label="Run notes（可选）"><input value={wandbNotes} onChange={(e) => setWandbNotes(e.target.value)} /></Field>
      </div> : null}
      {policy === "pi05" && !autodl && device.trim().toLowerCase() === "mps" ? <Banner tone="warn">pi0.5 不在 M1/MPS 上训练；启用 AutoDL 后会强制使用 CUDA。</Banner> : null}
      {!selectedDataset ? <Banner tone="warn">先在「训练集」页导出一个 LeRobotDataset。</Banner> : null}
      <div className="row"><button className="btn primary" disabled={!selectedDataset || Boolean(status?.running) || policy === "pi05" && !baseModel.trim() || policy === "pi05" && !autodl && device.trim().toLowerCase() === "mps" || autodl && !host.trim()} onClick={() => void start()}>{autodl ? "上传并在 AutoDL 训练" : "开始训练"}</button>
        {status?.running && (status.kind === "train" || status.kind === "autodl") ? <button className="btn danger" onClick={() => void api.stopTraining().then(setStatus)}>停止训练</button> : null}</div>
    </Card>
    <Card title="训练状态">
      {status?.kind === "train" || status?.kind === "autodl" ? <>
        <div className="stats"><Stat label="run" value={status.name || "—"} small /><Stat label="状态" value={status.phase} small /><Stat label="PID" value={status.pid || "—"} small /></div>
        {status.error ? <Banner tone="error">{status.error}</Banner> : null}
        <pre className="export-log">{status.log.join("\n") || "等待训练输出…"}</pre>
      </> : <Empty>训练状态会在这里显示。模型完成后出现在下方库中。</Empty>}
    </Card>
    <Card title="模型库" actions={<div className="row"><Pill>{models.length} 个</Pill><button className="btn danger small" disabled={!chosen.size || Boolean(status?.running)} onClick={() => void erase()}>删除所选 ({chosen.size})</button><button className="btn small" onClick={() => void reload()}>刷新</button></div>}>
      {!models.length ? <Empty>还没有模型。训练完成并写出 checkpoint 后会显示在这里。</Empty> : <div className="model-list">{models.map((model) => <label className="model-row" key={model.name}><input type="checkbox" checked={chosen.has(model.name)} onChange={() => toggle(model.name)} /><div><strong>{model.name}</strong><div className="dim mono">{model.ready ? "可执行" : "训练中 / checkpoint 未完成"} · {bytes(model.size_bytes)}</div></div><Pill tone={model.ready ? "ok" : "warn"}>{model.ready ? "READY" : "INCOMPLETE"}</Pill></label>)}</div>}
    </Card>
  </div>;
}
function bytes(value: number) { return value > 1e9 ? `${(value / 1e9).toFixed(2)} GB` : `${(value / 1e6).toFixed(1)} MB`; }
