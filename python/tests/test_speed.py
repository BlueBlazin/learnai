import time

import numpy as np
import pytest

from learnai.judge import CheckFailure, frameworks, speed


def busy(seconds):
    def run():
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pass

    return run


def test_compare_measures_the_ratio_of_medians():
    result = speed.compare(busy(0.004), busy(0.001), rounds=5, warmup=1)
    assert 3.0 < result.ratio < 5.0
    assert result.rounds == 5
    assert result.reference_s == pytest.approx(0.001, rel=0.3)


def test_calls_are_interleaved_and_warmed_up():
    order = []
    speed.compare(lambda: order.append("s"), lambda: order.append("r"), rounds=4, warmup=2)
    assert order[:4] == ["r", "s", "r", "s"]  # warmup, untimed
    timed = "".join(order[4:])
    # Both calibration and each round run both callables; neither is always second.
    assert "rs" in timed and "sr" in timed


def test_require_reports_the_metric_and_fails_beyond_the_factor(ctx):
    speed.require(speed.SpeedResult(0.002, 0.001, 7), max_ratio=3.0, label="f", ctx=ctx)
    assert ctx.metrics == [{"name": "slowdown", "value": 2, "unit": "x", "limit": 3}]
    with pytest.raises(CheckFailure) as failure:
        speed.require(speed.SpeedResult(0.010, 0.001, 7), max_ratio=3.0, label="f", ctx=ctx)
    assert failure.value.message == "f: 10.0x slower than the reference (limit 3x)"
    assert failure.value.detail == (
        "timing\n"
        "  solution     10 ms per call (median of 7)\n"
        "  reference    1 ms per call (median of 7)\n"
        "\n"
        "Timed in the same process, interleaved, after warmup.\n"
    )
    assert ctx.metrics[0]["value"] == 10


def test_lazy_frameworks_are_forced_before_the_clock_stops():
    try:
        fw = frameworks.get("mlx")
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))
    mx = fw.xp
    x = fw.tensor(np.ones((768, 768)))

    def work(n):
        y = x
        for _ in range(n):
            y = mx.tanh(y @ x / 768.0)
        return y

    # Without sync, MLX only builds a graph and both variants look alike. With sync
    # the 32x larger computation is measurably slower.
    result = speed.compare(lambda: fw.sync(work(64)), lambda: fw.sync(work(2)), rounds=5)
    assert result.ratio > 3.0
