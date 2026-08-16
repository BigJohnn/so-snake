import type { ReactNode } from "react";

export type Tone = "neutral" | "ok" | "warn" | "danger" | "accent";

export function Card({
  title,
  actions,
  children,
  padded = true
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
  padded?: boolean;
}) {
  return (
    <section className="card">
      <header>
        <h2>{title}</h2>
        {actions ? <div className="spacer">{actions}</div> : null}
      </header>
      {padded ? <div className="body">{children}</div> : children}
    </section>
  );
}

export function Pill({
  tone = "neutral",
  live = false,
  children
}: {
  tone?: Tone;
  live?: boolean;
  children: ReactNode;
}) {
  return (
    <span className={`pill ${tone === "neutral" ? "" : tone}`}>
      {tone !== "neutral" ? <span className={`dot${live ? " live" : ""}`} /> : null}
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  unit,
  small = false
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  small?: boolean;
}) {
  return (
    <div className="stat">
      <div className="k">{label}</div>
      <div className={`v${small ? " small" : ""}`}>
        {value}
        {unit ? <span className="u">{unit}</span> : null}
      </div>
    </div>
  );
}

export function Field({
  label,
  hint,
  children
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

/** A radio group that looks like one control, with per-option availability.
 *
 * Unavailable options stay visible and disabled rather than being hidden: an
 * operator whose controller is unplugged needs to see that "pro" exists and is
 * not currently possible, which a missing entry does not tell them. The reason
 * rides along as the title attribute. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  disabled = false
}: {
  value: T;
  options: { value: T; label: string; available?: boolean; reason?: string }[];
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="segmented">
      {options.map((option) => {
        const usable = option.available !== false;
        return (
          <button
            key={option.value}
            className={value === option.value ? "on" : ""}
            disabled={disabled || !usable}
            title={usable ? undefined : option.reason || "not available on this machine"}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export function Banner({
  tone,
  children
}: {
  // "ok" is not decoration. The export's verification is the one place in this
  // UI where a green result is a claim worth making -- "this dataset reads back
  // to what was recorded" -- and rendering it in the same grey as an ordinary
  // note would hide the only sentence the operator is waiting for.
  tone: "error" | "warn" | "info" | "ok";
  children: ReactNode;
}) {
  return <div className={`banner ${tone}`}>{children}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

/** A commanded-vs-measured bar within the joint's own limits.
 *
 * The command is the filled span from zero, the measurement is a separate tick.
 * Keeping them distinct is the point: the gap between them is the servo lag,
 * which is the single most useful thing to see while teleoperating -- if the
 * tick stops following the fill, the arm is stalled or holding. */
export function JointBar({
  name,
  commanded,
  measured,
  limits
}: {
  name: string;
  commanded: number;
  measured?: number;
  limits: [number, number];
}) {
  const [lo, hi] = limits;
  const span = Math.max(hi - lo, 1e-6);
  const pct = (v: number) => (100 * (Math.min(Math.max(v, lo), hi) - lo)) / span;
  const zero = pct(0);
  const cmd = pct(commanded);
  const left = Math.min(zero, cmd);
  const width = Math.abs(cmd - zero);

  return (
    <div className="joint">
      <div className="name">{name}</div>
      <div className="track">
        <div className="zero" style={{ left: `${zero}%` }} />
        <div className="fill" style={{ left: `${left}%`, width: `${width}%` }} />
        {measured === undefined ? null : (
          <div className="measured" style={{ left: `${pct(measured)}%` }} />
        )}
      </div>
      <div className="val">{commanded.toFixed(1)}°</div>
    </div>
  );
}
