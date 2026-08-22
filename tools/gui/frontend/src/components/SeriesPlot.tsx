import { useMemo, useRef, useState } from "react";

/* A small SVG time-series plot, written rather than pulled in.
 *
 * Every chart on this console is the same shape -- a few hundred samples of one
 * to five numeric series against time, refreshed a few times a second -- and a
 * charting library's value is in the cases this does not have: faceting, scales,
 * stacking, transitions. What it would cost is a dependency an order of
 * magnitude larger than the page, which for a tool whose whole point is that the
 * offline gates run without installing anything is the wrong trade.
 *
 * Colours are the validated dark-mode categorical slots, assigned by series
 * index in fixed order. Fixed order matters: the joint traces keep their colour
 * when a filter hides one, so `elbow_flex` is the same green in every plot. */

export const SERIES_COLORS = ["#197d7b", "#d75a39", "#238254", "#b47716", "#ad4f75"];

export interface Series {
  label: string;
  values: (number | null)[];
  /** Overrides the slot colour. Use only for a series with a fixed meaning. */
  color?: string;
}

export interface Band {
  /** Shaded where true — a state, not a series, so it is neutral and behind. */
  values: boolean[];
  label?: string;
}

interface Props {
  x: number[];
  series: Series[];
  height?: number;
  unit?: string;
  /** Pin the y-range. Leave open to fit the data, which is right for error traces. */
  yMin?: number;
  yMax?: number;
  /** A dashed horizontal marker — the nominal rate, a limit, a target. */
  reference?: { value: number; label: string };
  bands?: Band[];
  /** Shown when there is nothing to draw yet. */
  placeholder?: string;
  /** A pinned position, in sample index. Drawn distinctly from the hover line:
   *  this one is where the video is, and it stays put when the pointer leaves. */
  cursor?: number | null;
  /** Called with a sample index when the plot is clicked, so a click can seek
   *  the video the plot is aligned with. */
  onScrub?: (index: number) => void;
}

const PAD = { top: 8, right: 10, bottom: 16, left: 44 };

function niceTicks(lo: number, hi: number, count = 4): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [lo];
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(v);
  return ticks;
}

function format(value: number): string {
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 0.1) return value.toFixed(2);
  return value.toPrecision(2);
}

export function SeriesPlot({
  x,
  series,
  height = 110,
  unit,
  yMin,
  yMax,
  reference,
  bands,
  placeholder = "no data yet",
  cursor = null,
  onScrub
}: Props) {
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<number | null>(null);
  const ref = useRef<SVGSVGElement | null>(null);

  const measure = (node: SVGSVGElement | null) => {
    ref.current = node;
    if (node) {
      const w = node.getBoundingClientRect().width;
      if (w > 0 && Math.abs(w - width) > 1) setWidth(w);
    }
  };

  const domain = useMemo(() => {
    const finite: number[] = [];
    for (const s of series) for (const v of s.values) if (v !== null && Number.isFinite(v)) finite.push(v);
    if (reference) finite.push(reference.value);
    if (!finite.length) return null;
    let lo = yMin ?? Math.min(...finite);
    let hi = yMax ?? Math.max(...finite);
    if (hi - lo < 1e-9) {
      // A flat trace is information -- "the error is not moving" -- so give it a
      // band to sit in rather than dividing by zero or hiding it on an edge.
      const pad = Math.max(Math.abs(hi) * 0.1, 1e-6);
      lo -= pad;
      hi += pad;
    } else {
      const pad = (hi - lo) * 0.08;
      if (yMin === undefined) lo -= pad;
      if (yMax === undefined) hi += pad;
    }
    return { lo, hi };
  }, [series, reference, yMin, yMax]);

  /* Everything whose cost is proportional to the number of samples, computed
   * once per data change rather than once per render.
   *
   * The cursor moves at video frame rate -- the dataset page drives it from
   * `requestAnimationFrame` so the frame counter advances one frame at a time
   * instead of in the ~8-frame jumps `timeupdate` produces. That is 30 renders
   * a second of five plots, and rebuilding a 1200-point path string per series
   * on each of them is several milliseconds of string building competing with
   * two 1080p AV1 decodes. Memoised, a cursor move rebuilds one `<line>`.
   *
   * Deliberately above the empty-data early return: hooks cannot live after it.
   */
  const geometry = useMemo(() => {
    if (!x.length || !domain) return null;

    const plotW = Math.max(width - PAD.left - PAD.right, 10);
    const plotH = Math.max(height - PAD.top - PAD.bottom, 10);
    const n = x.length;
    const px = (i: number) => PAD.left + (n <= 1 ? plotW / 2 : (plotW * i) / (n - 1));
    const py = (v: number) => PAD.top + plotH * (1 - (v - domain.lo) / (domain.hi - domain.lo));

    const path = (values: (number | null)[]) => {
      let d = "";
      let pen = false;
      values.forEach((v, i) => {
        if (v === null || !Number.isFinite(v)) {
          pen = false;
          return;
        }
        d += `${pen ? "L" : "M"}${px(i).toFixed(1)},${py(v).toFixed(1)}`;
        pen = true;
      });
      return d;
    };

    const bandRects = (values: boolean[]) => {
      const rects: { x: number; w: number }[] = [];
      let start: number | null = null;
      values.forEach((on, i) => {
        if (on && start === null) start = i;
        if ((!on || i === values.length - 1) && start !== null) {
          const end = on ? i : i - 1;
          rects.push({ x: px(start), w: Math.max(px(end) - px(start), 1) });
          start = null;
        }
      });
      return rects;
    };

    return {
      plotW,
      plotH,
      n,
      px,
      py,
      paths: series.map((s) => path(s.values)),
      rects: (bands ?? []).map((band) => bandRects(band.values)),
      ticks: niceTicks(domain.lo, domain.hi, 3)
    };
  }, [x, series, bands, width, height, domain]);

  if (!x.length || !domain || !geometry) {
    return (
      <svg ref={measure} className="plot" height={height} role="img" aria-label={placeholder}>
        <text x="50%" y="50%" textAnchor="middle" fill="var(--dim)" fontSize="11" fontFamily="var(--mono)">
          {placeholder}
        </text>
      </svg>
    );
  }

  const { plotW, plotH, n, px, py, ticks } = geometry;
  const multi = series.length > 1;

  return (
    <div>
      {multi ? (
        <div className="plot-title">
          <span>
            {series.map((s, i) => (
              <span key={s.label} style={{ marginRight: 12 }}>
                <span
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 2,
                    background: s.color ?? SERIES_COLORS[i % SERIES_COLORS.length],
                    verticalAlign: "middle",
                    marginRight: 5
                  }}
                />
                {s.label}
              </span>
            ))}
          </span>
          {unit ? <span>{unit}</span> : null}
        </div>
      ) : (
        <div className="plot-title">
          <span>{series[0]?.label}</span>
          {unit ? <span>{unit}</span> : null}
        </div>
      )}

      <svg
        ref={measure}
        className="plot"
        height={height}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const i = Math.round(((event.clientX - rect.left - PAD.left) / plotW) * (n - 1));
          setHover(i >= 0 && i < n ? i : null);
        }}
        onMouseLeave={() => setHover(null)}
        onClick={
          onScrub
            ? (event) => {
                const rect = event.currentTarget.getBoundingClientRect();
                const i = Math.round(((event.clientX - rect.left - PAD.left) / plotW) * (n - 1));
                if (i >= 0 && i < n) onScrub(i);
              }
            : undefined
        }
        style={onScrub ? { cursor: "pointer" } : undefined}
      >
        {geometry.rects.flatMap((band, bi) =>
          band.map((rect, ri) => (
            <rect
              key={`b${bi}-${ri}`}
              x={rect.x}
              y={PAD.top}
              width={rect.w}
              height={plotH}
              fill="var(--accent)"
              opacity={0.09}
            />
          ))
        )}

        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={PAD.left + plotW}
              y1={py(tick)}
              y2={py(tick)}
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text x={PAD.left - 6} y={py(tick) + 3.5} textAnchor="end" fontSize="10" fill="var(--dim)" fontFamily="var(--mono)">
              {format(tick)}
            </text>
          </g>
        ))}

        {reference ? (
          <g>
            <line
              x1={PAD.left}
              x2={PAD.left + plotW}
              y1={py(reference.value)}
              y2={py(reference.value)}
              stroke="var(--muted)"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
            <text x={PAD.left + plotW} y={py(reference.value) - 4} textAnchor="end" fontSize="10" fill="var(--muted)" fontFamily="var(--mono)">
              {reference.label}
            </text>
          </g>
        ) : null}

        {series.map((s, i) => (
          <path
            key={s.label}
            d={geometry.paths[i]}
            fill="none"
            stroke={s.color ?? SERIES_COLORS[i % SERIES_COLORS.length]}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}

        {cursor !== null && cursor >= 0 && cursor < x.length ? (
          <line
            x1={px(cursor)}
            x2={px(cursor)}
            y1={PAD.top}
            y2={PAD.top + plotH}
            stroke="var(--accent)"
            strokeWidth={1.5}
          />
        ) : null}
        {hover !== null ? (
          <g>
            <line x1={px(hover)} x2={px(hover)} y1={PAD.top} y2={PAD.top + plotH} stroke="var(--muted)" strokeWidth={1} />
            {series.map((s, i) => {
              const v = s.values[hover];
              if (v === null || v === undefined || !Number.isFinite(v)) return null;
              return (
                <circle
                  key={s.label}
                  cx={px(hover)}
                  cy={py(v)}
                  r={3.5}
                  fill={s.color ?? SERIES_COLORS[i % SERIES_COLORS.length]}
                  stroke="var(--panel)"
                  strokeWidth={2}
                />
              );
            })}
          </g>
        ) : null}

        <text x={PAD.left} y={height - 3} fontSize="10" fill="var(--dim)" fontFamily="var(--mono)">
          {format(x[0])}s
        </text>
        <text x={PAD.left + plotW} y={height - 3} textAnchor="end" fontSize="10" fill="var(--dim)" fontFamily="var(--mono)">
          {format(x[n - 1])}s
        </text>
      </svg>

      {hover !== null ? (
        <div className="plot-title" style={{ marginTop: 2 }}>
          <span>t = {format(x[hover])} s</span>
          <span>
            {series
              .map((s) => {
                const v = s.values[hover];
                return v === null || v === undefined ? null : `${s.label} ${format(v)}`;
              })
              .filter(Boolean)
              .join("   ")}
          </span>
        </div>
      ) : null}
    </div>
  );
}
