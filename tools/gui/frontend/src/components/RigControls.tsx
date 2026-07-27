import type { RigBody } from "../api";
import { Banner, Field, Segmented } from "./ui";
import type { AppConfig, BackendKind, SourceKind } from "../types";

export interface RigState {
  backend: BackendKind;
  source: SourceKind;
  port: string;
  maxRelativeTarget: number;
}

export const DEFAULT_RIG: RigState = {
  backend: "mujoco",
  source: "scripted",
  port: "",
  maxRelativeTarget: 5
};

export function rigBody(rig: RigState): RigBody {
  return {
    backend: rig.backend,
    source: rig.source,
    port: rig.port,
    max_relative_target_deg: rig.maxRelativeTarget
  };
}

/** Whether the operator has supplied everything the chosen rig needs. */
export function rigReady(rig: RigState, config: AppConfig): string {
  const backend = config.availability.backends[rig.backend];
  if (!backend?.available) return backend?.reason || `${rig.backend} 不可用`;
  if (rig.backend === "real" && !rig.port.trim()) return "真机需要串口";
  return "";
}

export function RigControls({
  rig,
  onChange,
  config,
  disabled,
  showSource = true
}: {
  rig: RigState;
  onChange: (rig: RigState) => void;
  config: AppConfig;
  disabled: boolean;
  showSource?: boolean;
}) {
  const backends = config.availability.backends;
  const sources = config.availability.sources;

  return (
    <>
      <Field label="机械臂 (backend)" hint="mock 无仿真; mujoco 带网格干涉检查; real 驱动真臂">
        <Segmented<BackendKind>
          value={rig.backend}
          disabled={disabled}
          onChange={(backend) => onChange({ ...rig, backend })}
          options={[
            { value: "mock", label: "mock", available: backends.mock?.available, reason: backends.mock?.reason },
            { value: "mujoco", label: "mujoco", available: backends.mujoco?.available, reason: backends.mujoco?.reason },
            { value: "real", label: "real ⚠", available: backends.real?.available, reason: backends.real?.reason }
          ]}
        />
      </Field>

      {showSource ? (
        <Field label="输入 (source)" hint="scripted 为内置波形; pro 为 Switch Pro 手柄 (按住 ZL 才动)">
          <Segmented<SourceKind>
            value={rig.source}
            disabled={disabled}
            onChange={(source) => onChange({ ...rig, source })}
            options={[
              { value: "scripted", label: "scripted", available: sources.scripted?.available, reason: sources.scripted?.reason },
              { value: "pro", label: "pro 手柄", available: sources.pro?.available, reason: sources.pro?.reason }
            ]}
          />
        </Field>
      ) : null}

      {rig.backend === "real" ? (
        <>
          <Banner tone="warn">
            真机模式会给舵机上力矩并驱动机械臂。先跑 <span className="mono">scripts/preflight_real_arm.py</span>,
            清空工作区,手放急停旁。
          </Banner>
          <Field label="串口" hint="Linux 一般是 /dev/ttyACM0, macOS 是 /dev/cu.usbmodem*">
            <input
              value={rig.port}
              disabled={disabled}
              placeholder="/dev/ttyACM0"
              onChange={(event) => onChange({ ...rig, port: event.target.value })}
            />
          </Field>
          <Field
            label={`硬件每步钳位 ${rig.maxRelativeTarget.toFixed(1)}°`}
            hint="lerobot 硬件层限幅,叠加在回路自身的 6°/步 之上。首次上机取小值。"
          >
            <input
              type="range"
              min={1}
              max={15}
              step={0.5}
              value={rig.maxRelativeTarget}
              disabled={disabled}
              onChange={(event) => onChange({ ...rig, maxRelativeTarget: Number(event.target.value) })}
            />
          </Field>
          {!config.availability.joint_map_present ? (
            <Banner tone="error">
              缺少关节映射 <span className="mono">{config.availability.joint_map_path}</span>,
              先跑 <span className="mono">scripts/map_joint_frames.py</span>。
            </Banner>
          ) : null}
        </>
      ) : null}
    </>
  );
}
