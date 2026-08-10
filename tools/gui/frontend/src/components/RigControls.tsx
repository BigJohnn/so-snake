import { useState } from "react";
import { api, ApiError, type RigBody } from "../api";
import { Banner, Field, Segmented } from "./ui";
import type { AppConfig, BackendKind, CameraDevice, CameraRole, SourceKind } from "../types";

export interface RigState {
  backend: BackendKind;
  source: SourceKind;
  port: string;
  maxRelativeTarget: number;
  /** Role -> device index, as a string because it comes out of a <select>. */
  cameras: Record<CameraRole, string>;
}

export const DEFAULT_RIG: RigState = {
  backend: "mujoco",
  source: "scripted",
  port: "",
  maxRelativeTarget: 5,
  // No camera is opened until the operator picks one. Guessing would mean
  // opening whatever device happens to be index 0, which on a laptop is the
  // built-in webcam pointed at the operator rather than at the workspace.
  cameras: { third_person: "", wrist: "" }
};

export function rigBody(rig: RigState): RigBody {
  const cameras: Record<string, number | string> = {};
  for (const [role, device] of Object.entries(rig.cameras)) {
    if (device !== "") cameras[role] = device;
  }
  return {
    backend: rig.backend,
    source: rig.source,
    port: rig.port,
    max_relative_target_deg: rig.maxRelativeTarget,
    cameras
  };
}

/** Whether the operator has supplied everything the chosen rig needs. */
export function rigReady(rig: RigState, config: AppConfig): string {
  const backend = config.availability.backends[rig.backend];
  if (!backend?.available) return backend?.reason || `${rig.backend} 不可用`;
  if (rig.backend === "real" && !rig.port.trim()) return "真机需要串口";
  // One device cannot feed two roles: the second `connect` on the same index
  // fails, and the failure lands in the worker after the session has started.
  const picked = Object.values(rig.cameras).filter((d) => d !== "");
  if (new Set(picked).size !== picked.length) return "两个视角选了同一个相机";
  return "";
}

const ROLE_LABELS: Record<CameraRole, string> = {
  third_person: "第三人称",
  wrist: "腕部"
};

/** Device pickers for the two camera roles, with an on-demand scan.
 *
 * Scanning is a button rather than something that happens on mount because it
 * opens every camera on the machine to see which ones deliver a frame -- that
 * takes seconds, and on macOS the first one triggers the system permission
 * prompt. Neither belongs in a page load.
 */
function CameraPickers({
  rig,
  onChange,
  config,
  disabled
}: {
  rig: RigState;
  onChange: (rig: RigState) => void;
  config: AppConfig;
  disabled: boolean;
}) {
  const [scanned, setScanned] = useState<CameraDevice[] | null>(null);
  const [scanning, setScanning] = useState(false);
  const [error, setError] = useState("");
  // Hidden by the operator, remembered per browser. This is the only dependable
  // way to get the built-in webcam and OBS's virtual camera out of the way on
  // macOS: the platform can say which devices are USB, but not which OpenCV
  // index each of them is, so a "USB only" filter would be guessing about
  // exactly the thing that must not be guessed. Hiding by hand cannot be wrong
  // -- the operator is looking at the picture while they do it.
  const [hidden, setHidden] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("so-snake.hidden-cameras") ?? "[]");
    } catch {
      return [];
    }
  });
  const [showHidden, setShowHidden] = useState(false);

  const hide = (key: string, off: boolean) => {
    const next = off ? [...new Set([...hidden, key])] : hidden.filter((k) => k !== key);
    setHidden(next);
    try {
      localStorage.setItem("so-snake.hidden-cameras", JSON.stringify(next));
    } catch {
      /* a browser that refuses storage still gets the filter for this session */
    }
  };

  const devices = scanned?.filter((d) => showHidden || !hidden.includes(String(d.device))) ?? null;
  const hiddenCount = (scanned?.length ?? 0) - (devices?.length ?? 0);

  const availability = config.availability.cameras;
  if (!availability?.available) {
    return (
      <Field label="相机">
        <Banner tone="info">相机不可用:{availability?.reason || "缺少依赖"}</Banner>
      </Field>
    );
  }

  const scan = async () => {
    setScanning(true);
    setError("");
    try {
      setScanned(await api.cameras());
    } catch (exc) {
      setError(exc instanceof ApiError ? exc.message : String(exc));
    } finally {
      setScanning(false);
    }
  };

  const setRole = (role: CameraRole, device: string) =>
    onChange({ ...rig, cameras: { ...rig.cameras, [role]: device } });

  return (
    <Field
      label="相机"
      hint="按画面认相机,不要按编号 —— macOS 上编号既不对应设备名,也不在重插后保持不变"
    >
      <div className="row">
        <button className="btn" disabled={disabled || scanning} onClick={() => void scan()}>
          {scanning ? "扫描中…" : scanned ? "重新扫描" : "扫描相机"}
        </button>
        {hiddenCount > 0 || showHidden ? (
          <button className="btn small" onClick={() => setShowHidden(!showHidden)}>
            {showHidden ? "收起隐藏的" : `显示隐藏的 (${hiddenCount})`}
          </button>
        ) : null}
      </div>

      {error ? <Banner tone="error">{error}</Banner> : null}
      {devices !== null && devices.length === 0 && !error ? (
        <Banner tone="warn">
          没扫到相机。macOS 上先确认终端有摄像头权限(系统设置 → 隐私与安全性 → 摄像头)。
        </Banner>
      ) : null}

      {(Object.keys(ROLE_LABELS) as CameraRole[]).map((role) => (
        <div className="camera-pick" key={role}>
          <div className="row">
            <span className="mono">{ROLE_LABELS[role]}</span>
            {rig.cameras[role] !== "" ? (
              <button className="btn small" disabled={disabled} onClick={() => setRole(role, "")}>
                清除
              </button>
            ) : null}
          </div>
          {devices?.length ? (
            // Thumbnails rather than a dropdown of names: on macOS there is no
            // name to show, and two identical cameras are only distinguishable
            // by what they are pointed at.
            <div className="camera-choices">
              {devices.map((device) => {
                const value = String(device.device);
                const picked = rig.cameras[role] === value;
                const isHidden = hidden.includes(value);
                return (
                  <div className={`camera-choice-wrap${isHidden ? " dimmed" : ""}`} key={device.index}>
                    <button
                      className={`camera-choice${picked ? " picked" : ""}`}
                      disabled={disabled}
                      onClick={() => setRole(role, picked ? "" : value)}
                      title={
                        `${device.width}×${device.height}\n${device.device}\n` +
                        (device.stable
                          ? "重插后仍指向同一台相机"
                          : "编号会随设备增减变化,重插后需重扫")
                      }
                    >
                      {device.thumbnail ? (
                        <img src={device.thumbnail} alt={`相机 ${device.index}`} />
                      ) : (
                        <div className="camera-choice-blank" />
                      )}
                      <span className="mono">
                        {device.stable ? "🔒" : ""}
                        {device.bus === "usb" ? "USB " : ""}
                        {device.name || device.index}
                      </span>
                    </button>
                    <button
                      className="camera-hide"
                      title={isHidden ? "取消隐藏" : "以后不再列出这个设备"}
                      onClick={() => hide(value, !isHidden)}
                    >
                      {isHidden ? "↩" : "×"}
                    </button>
                  </div>
                );
              })}
            </div>
          ) : (
            <span className="hint">先扫描</span>
          )}
        </div>
      ))}
    </Field>
  );
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

      <CameraPickers rig={rig} onChange={onChange} config={config} disabled={disabled} />
    </>
  );
}
