import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "./api";
import type { SeriesRow, Snapshot } from "./types";

/** Poll a function on an interval, skipping ticks while one is still in flight.
 *
 * Chaining the next timer off the previous response, rather than using
 * `setInterval`, is what keeps a slow or wedged gateway from queueing up
 * requests behind itself -- at 10 Hz that turns one stall into a pile-up that
 * outlives the stall. */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number,
  enabled = true
): { data: T | null; error: string; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [nonce, setNonce] = useState(0);
  const alive = useRef(true);
  const saved = useRef(fetcher);
  saved.current = fetcher;

  const refresh = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    alive.current = true;
    if (!enabled) return () => undefined;
    let timer: number | undefined;

    const tick = async () => {
      try {
        const value = await saved.current();
        if (!alive.current) return;
        setData(value);
        setError("");
      } catch (cause) {
        if (!alive.current) return;
        setError(cause instanceof ApiError ? cause.message : String(cause));
      }
      if (alive.current) timer = window.setTimeout(tick, intervalMs);
    };

    void tick();
    return () => {
      alive.current = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [intervalMs, enabled, nonce]);

  return { data, error, refresh };
}

/** The session snapshot, polled fast while something is moving and slow when not.
 *
 * The rate is not cosmetic: at 30 Hz control, a 1 Hz poll shows an operator a
 * clutch state a second out of date, and they will act on it. Idle, there is
 * nothing to be late about. */
export function useSnapshot(): {
  snapshot: Snapshot | null;
  error: string;
  refresh: () => void;
} {
  const [busy, setBusy] = useState(false);
  const { data, error, refresh } = usePolling<Snapshot>(api.snapshot, busy ? 100 : 1000);

  useEffect(() => {
    if (data) setBusy(data.mode !== "idle");
  }, [data]);

  return { snapshot: data, error, refresh };
}

export function useSeries(enabled: boolean): SeriesRow[] {
  const { data } = usePolling<SeriesRow[]>(() => api.series(600), 500, enabled);
  return data ?? [];
}

/** A counter that ticks while `enabled`, used to bust the preview image's cache. */
export function useTicker(intervalMs: number, enabled: boolean): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!enabled) return () => undefined;
    const timer = window.setInterval(() => setTick((t) => t + 1), intervalMs);
    return () => window.clearInterval(timer);
  }, [intervalMs, enabled]);
  return tick;
}
