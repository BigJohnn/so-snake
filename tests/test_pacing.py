"""`RateKeeper`: the thing that makes a configured rate the achieved rate.

These are wall-clock tests, which is unusual here and unavoidable: the bug they
pin is *in* the wall clock. `time.sleep` returns late by an amount that scales
with the requested sleep (~4 ms on a 33 ms period on this bench), and the loop
that trusted it recorded 43 episodes at 26 Hz against a configured 30. Nothing
short of measuring elapsed time can show that.

They are kept cheap -- fractions of a second each -- and their tolerances are
wide enough to survive a loaded CI machine while still being far tighter than
the 12% error they exist to catch.
"""

from __future__ import annotations

import statistics
import time

import pytest

from so_snake.pacing import SPIN_MARGIN_S, RateKeeper


def test_it_holds_the_rate_that_sleep_alone_misses() -> None:
    hz = 50.0
    steps = 25  # half a second

    keeper = RateKeeper(hz)
    start = time.perf_counter()
    for _ in range(steps):
        keeper.wait()
    measured = steps / (time.perf_counter() - start)

    assert measured == pytest.approx(hz, rel=0.05), f"{measured:.2f} Hz vs {hz} Hz"


def test_sleeping_for_the_whole_period_would_not_have() -> None:
    """The premise, measured rather than asserted about.

    `RateKeeper` only earns its spin if `time.sleep` really does return late. A
    test that checked the fix alone would keep passing on a platform where the
    premise had stopped holding, and nobody would notice the margin had become
    dead weight.
    """
    hz = 50.0
    period = 1.0 / hz
    steps = 25

    start = time.perf_counter()
    for _ in range(steps):
        time.sleep(period)
    naive = steps / (time.perf_counter() - start)

    assert naive < hz, (
        f"sleep({period * 1000:.0f} ms) held {naive:.2f} Hz against a requested "
        f"{hz} Hz. If this platform's timers are exact, RateKeeper's spin margin "
        "is no longer buying anything."
    )


def test_a_late_step_is_repaid_by_the_next_one() -> None:
    """Pacing is against an advancing deadline, not the top of the iteration.

    `period - elapsed` books every overrun permanently, so a loop can only fall
    behind. Here one step overruns and the following waits are shorter, so the
    grid comes back.
    """
    hz = 50.0
    period = 1.0 / hz
    keeper = RateKeeper(hz)

    start = time.perf_counter()
    for i in range(20):
        if i == 5:
            time.sleep(period)  # a step that took a whole extra period
        keeper.wait()
    elapsed = time.perf_counter() - start

    # 20 periods, not 21: the overrun is absorbed rather than added on.
    assert elapsed == pytest.approx(20 * period, rel=0.12)


def test_a_long_stall_is_written_off_not_repaid() -> None:
    """Repayment stops at one period, so a stall cannot buy a burst.

    A step that blocked for far longer than a period -- a USB stall; this bench
    has seen 400 ms -- would otherwise leave the deadline so far in the past
    that the next dozen steps run with no wait at all, driving the servo bus
    faster than the arm was ever commanded at.
    """
    hz = 100.0
    period = 1.0 / hz
    keeper = RateKeeper(hz)

    time.sleep(10 * period)
    keeper.wait()

    # The next waits must be real waits, not a free run through stale deadlines.
    start = time.perf_counter()
    for _ in range(10):
        keeper.wait()
    elapsed = time.perf_counter() - start

    assert elapsed == pytest.approx(10 * period, rel=0.25)


def test_disabled_costs_nothing() -> None:
    """Offline runs must not pay for pacing; every gate here runs that way."""
    keeper = RateKeeper(1.0, enabled=False)

    start = time.perf_counter()
    for _ in range(100):
        keeper.wait()

    assert time.perf_counter() - start < 0.05


def test_wait_reports_the_slack_it_had() -> None:
    keeper = RateKeeper(20.0)

    assert keeper.wait() == pytest.approx(1.0 / 20.0, abs=0.01)

    time.sleep(1.0 / 20.0 + 0.01)
    slack = keeper.wait()
    assert slack < 0.0, "an overrun step must report negative slack"
    assert keeper.overruns == 1


def test_a_nonpositive_rate_is_refused() -> None:
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            RateKeeper(bad)


def test_the_spin_margin_covers_the_platforms_overshoot() -> None:
    """The margin has to be bigger than what `sleep` adds, or it does nothing.

    Measured rather than assumed: the design rests on the overshoot being
    bounded and small, and on a platform that overshoots by more than the margin
    the spin never runs and the rate quietly drops again.

    The *typical* overshoot, not the worst. A loaded machine -- this suite
    running in parallel is enough -- can delay any single wake-up arbitrarily,
    and no fixed margin covers that. The margin does not have to: an occasional
    late step is repaid by the next one's shorter wait, which is the whole point
    of pacing to an advancing deadline. What must hold is that the ordinary case
    is covered, or every step is late and there is nothing to repay from.
    """
    request = 0.02
    overshoots = []
    for _ in range(15):
        start = time.perf_counter()
        time.sleep(request)
        overshoots.append(time.perf_counter() - start - request)
    typical = statistics.median(overshoots)

    assert typical < SPIN_MARGIN_S, (
        f"sleep({request * 1000:.0f} ms) typically overshot by {typical * 1000:.1f} ms, "
        f"more than the {SPIN_MARGIN_S * 1000:.0f} ms spin margin "
        f"(worst of {len(overshoots)}: {max(overshoots) * 1000:.1f} ms)"
    )
