"""Holding a fixed control rate, which `time.sleep` alone does not do.

Every loop in this repository that drives the arm at a configured rate -- the
teleoperation loop, the replayer -- paces itself through `RateKeeper`. It is one
class rather than four lines copied twice because both of the mistakes it avoids
are silent, and both of them corrupt recorded data rather than crashing.

## `time.sleep` returns late, by more than enough to matter

macOS coalesces timer wake-ups: it rounds a sleeping thread's deadline outwards
so several threads can be woken together and the cores can stay asleep longer.
The slack scales with the requested sleep. Measured on this bench (M1 Pro):

    requested    returned    overshoot
      1.0 ms      1.46 ms     +0.5 ms
      5.0 ms      6.19 ms     +1.2 ms
     28.3 ms     32.42 ms     +4.1 ms
     33.3 ms     37.32 ms     +4.0 ms

A 30 Hz loop whose step costs 5 ms asks to sleep for the remaining 28 ms and
gets 32 back, so it runs at 26.7 Hz. That is the entire story of why this bench
recorded 43 episodes at 26 Hz against a configured 30: not a slow step -- a step
is about 5 ms of a 33 ms budget -- but a sleep that could not be asked to be
shorter. So the last few milliseconds of each period are spun, not slept.

## Pacing from the top of the iteration books every overshoot permanently

`sleep(period - elapsed)` measured from the start of the current step treats
lost time as spent. The loop can only fall behind, never catch up, and the
shortfall compounds across a take. A deadline that advances by exactly one
period instead lets a late step be repaid by the next one's shorter wait.

Repayment is capped at one period. Beyond that the loop is not late, it was
blocked -- a USB stall, a GC pause; this bench has seen 400 ms -- and a deadline
that far in the past would otherwise buy a burst of steps with no wait at all,
driving the servo bus faster than the arm was ever commanded at. Past the cap
the lost time is written off and the grid restarts.
"""

from __future__ import annotations

import time

# How much of each period's wait is spun rather than slept. Large enough to
# cover the measured overshoot above with headroom; at 30 Hz it costs at most
# 6 ms of one core per 33 ms period. That is real CPU, and it buys a dataset
# whose rows sit on the grid its timestamps claim.
SPIN_MARGIN_S = 0.006


class RateKeeper:
    """Paces a loop to `hz`, absorbing `time.sleep`'s overshoot.

    Call `wait()` once per iteration, at the bottom::

        keeper = RateKeeper(30.0)
        while running:
            do_one_step()
            keeper.wait()

    `enabled=False` makes `wait()` a no-op, for offline runs where pacing to
    the wall clock only makes a test slow.
    """

    def __init__(self, hz: float, *, enabled: bool = True, now: float | None = None) -> None:
        if hz <= 0.0:
            raise ValueError(f"hz must be positive, got {hz}")
        self.period = 1.0 / float(hz)
        self.enabled = bool(enabled)
        self._deadline = time.perf_counter() if now is None else float(now)
        self.overruns = 0

    @property
    def hz(self) -> float:
        return 1.0 / self.period

    def reset(self) -> None:
        """Restart the grid from now. For a loop resuming after a known pause."""
        self._deadline = time.perf_counter()

    def wait(self) -> float:
        """Sleep and spin until this iteration's deadline. Returns the slack.

        The slack is what was left when the deadline was computed: negative
        means the step overran its budget, which is the number worth logging
        when a loop is not holding its rate.
        """
        if not self.enabled:
            return 0.0

        self._deadline += self.period
        slack = self._deadline - time.perf_counter()
        if slack > SPIN_MARGIN_S:
            time.sleep(slack - SPIN_MARGIN_S)
        # The tail, spun: this is what actually holds the rate, because the
        # sleep above returns late by an amount sleeping cannot correct for.
        while time.perf_counter() < self._deadline:
            pass

        if slack < 0.0:
            self.overruns += 1
            # Repay at most one period; see the module docstring.
            if -slack > self.period:
                self._deadline = time.perf_counter()
        return slack
