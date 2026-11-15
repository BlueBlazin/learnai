"""Speed judge: relative timing against the same-framework reference.

Absolute wall-clock thresholds are meaningless across machines, so the only question
asked is "how much slower than the reference, here, now?". Both callables run in this
process, alternating, so machine load, thermal state, and caches affect both alike.
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from whetstone.judge.errors import CheckFailure

#: Each timed measurement repeats its call until it lasts about this long, so that
#: timer resolution and per-call jitter do not dominate fast functions.
MIN_MEASURE_S = 0.005
MAX_REPEATS = 2000


@dataclass(frozen=True)
class SpeedResult:
    solution_s: float  # median seconds per call
    reference_s: float
    rounds: int

    @property
    def ratio(self) -> float:
        return self.solution_s / self.reference_s if self.reference_s > 0 else math.inf


def _time(fn: Callable[[], Any], repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def _repeats_for(once: float) -> int:
    return max(1, min(MAX_REPEATS, math.ceil(MIN_MEASURE_S / max(once, 1e-9))))


def compare(
    run_solution: Callable[[], Any],
    run_reference: Callable[[], Any],
    *,
    rounds: int = 7,
    warmup: int = 2,
) -> SpeedResult:
    """Time two zero-argument callables against each other.

    Each callable must force its own result (``fw.sync``) before returning; otherwise
    a lazy framework is timed building a graph instead of computing. Procedure:
    ``warmup`` untimed calls each (allocator, caches, lazy initialization), then
    ``rounds`` rounds; every round measures both, in alternating order so that neither
    always runs second; the result is the median over rounds for each.

    Before the rounds, each callable is timed once to decide how often a measurement
    repeats it. When neither needs repeating (one call already lasts ``MIN_MEASURE_S``),
    that pair of timings *is* a round measured exactly like the others (after
    warm-up, garbage collection off, reference first as in every even round), so it
    counts as the first one instead of being thrown away: slow callables are not run
    once more than the statistics need.
    """
    for _ in range(warmup):
        run_reference()
        run_solution()
    solution_times: list[float] = []
    reference_times: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        reference_once = _time(run_reference, 1)
        solution_once = _time(run_solution, 1)
        reference_repeats = _repeats_for(reference_once)
        solution_repeats = _repeats_for(solution_once)
        if reference_repeats == 1 and solution_repeats == 1 and rounds >= 1:
            reference_times.append(reference_once)
            solution_times.append(solution_once)
        for round_index in range(len(solution_times), rounds):
            if round_index % 2 == 0:
                reference_times.append(_time(run_reference, reference_repeats))
                solution_times.append(_time(run_solution, solution_repeats))
            else:
                solution_times.append(_time(run_solution, solution_repeats))
                reference_times.append(_time(run_reference, reference_repeats))
    finally:
        if gc_was_enabled:
            gc.enable()
    return SpeedResult(
        solution_s=statistics.median(solution_times),
        reference_s=statistics.median(reference_times),
        rounds=rounds,
    )


def require(
    result: SpeedResult,
    *,
    max_ratio: float,
    label: str,
    ctx: Any = None,
    tokens: int | None = None,
) -> None:
    """Report the ``slowdown`` metric and fail if it exceeds ``max_ratio``.

    With ``tokens`` (how many tokens one call handles) the solution's throughput is
    reported too, as ``tokens_per_s``: a number to look at, never a threshold.
    """
    if ctx is not None:
        ctx.metric("slowdown", round(result.ratio, 3), "x", max_ratio)
        if tokens is not None and result.solution_s > 0:
            rate = tokens / result.solution_s
            ctx.metric("tokens_per_s", round(rate, 1), "tok/s", higher_is_better=True)
    if result.ratio > max_ratio:
        raise CheckFailure(
            f"{label}: {result.ratio:.1f}x slower than the reference (limit {max_ratio:g}x)",
            f"solution:  {result.solution_s * 1e3:.4g} ms per call (median of {result.rounds})\n"
            f"reference: {result.reference_s * 1e3:.4g} ms per call (median of {result.rounds})\n"
            "Timed in the same process, interleaved, after warmup.\n",
        )
