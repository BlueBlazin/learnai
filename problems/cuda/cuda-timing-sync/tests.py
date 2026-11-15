"""Tests for cuda-timing-sync.

The CPU cases use the kit's `FakeDevice`: a device that works asynchronously on a
virtual clock kept in integer nanoseconds. A function made by `dev.kernel(seconds)`
queues that much device work and returns at once; `dev.sync()` moves host time to the
moment the queue is empty; `dev.clock()` reads host time and never waits. Durations
are therefore exact, and each timing mistake has an exact wrong value:

- no sync before the stop: 0 (only the launch is timed);
- no sync before the start: the first sample includes everything queued earlier;
- no warm-up: the first sample is the slow first call.

A median hides one bad sample, so the last two are tested with n = 1 and n = 2.

The GPU case (`requires="cuda"`) was written from the PyTorch 2.15 documentation and
has never run on the machine that validates the catalog.
"""

import statistics
import time

import torch

from whetstone.judge import case, check
from whetstone.kit import FakeDevice, RealDevice
from whetstone.kit.counters import Counters

FN = "time_region"


def counted(fn):
    """`fn` with a counter of its calls."""
    counters = Counters("fn_calls")

    def call():
        counters.add("fn_calls")
        return fn()

    return call, counters


def scripted(dev, durations):
    """A function whose k-th call queues durations[k] seconds of device work.

    Calls beyond the script repeat the last duration; the call count is checked later.
    """
    remaining = list(durations)

    def fn():
        dev.launch(remaining.pop(0) if len(remaining) > 1 else remaining[0])

    return fn


def measure(ctx, dev, fn, n, expected):
    """Run the solution; check the value, then the number of calls."""
    call, counters = counted(fn)
    seconds = ctx.solution.time_region(call, dev, n)
    check.that(
        isinstance(seconds, float),
        f"{FN}: returned a {type(seconds).__name__}, not a Python float",
    )
    check.allclose(seconds, expected, rtol=1e-9, atol=1e-12, label="median seconds per call")
    check.cost(counters, "fn_calls", exactly=n + 1, unit="calls")
    return seconds


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: 4 ms per call after a 50 ms first call, n = 5", public=True)
def worked_example(ctx):
    # Rejects no_sync_before_stop and sync_after_clock (they measure 0), host_clock
    # (a different clock), and warmup_every_sample (10 calls).
    dev = FakeDevice()
    fn = dev.kernel(0.004, first_call_s=0.050)
    measure(ctx, dev, fn, 5, 0.004)


@case("work was already queued when time_region is called, n = 1", public=True)
def stale_work(ctx):
    # Rejects no_sync_before_start: its only sample is 0.100 + 0.050 + 0.004.
    dev = FakeDevice()
    fn = dev.kernel(0.004, first_call_s=0.050)
    dev.launch(0.100)  # someone else's work, still running
    measure(ctx, dev, fn, 1, 0.004)


@case("the first call is slow: it is the warm-up and is not a sample, n = 1 and n = 2", public=True)
def warmup_excluded(ctx):
    # Rejects no_warmup: 0.050 for n = 1, (0.050 + 0.004) / 2 for n = 2.
    dev = FakeDevice()
    measure(ctx, dev, dev.kernel(0.004, first_call_s=0.050), 1, 0.004)
    dev = FakeDevice()
    measure(ctx, dev, dev.kernel(0.004, first_call_s=0.050), 2, 0.004)


@case("calls of different durations: the median, not the mean", public=True)
def median_of_samples(ctx):
    # Rejects mean_not_median and total_over_n (0.008) and unsorted_median (0.020).
    dev = FakeDevice()
    fn = scripted(dev, [0.030, 0.003, 0.001, 0.020, 0.002, 0.014])  # the first is the warm-up
    measure(ctx, dev, fn, 5, 0.003)


@case("an even number of samples: the mean of the two middle durations")
def even_count(ctx):
    # Rejects upper_median (0.006).
    dev = FakeDevice()
    fn = scripted(dev, [0.010, 0.006, 0.001, 0.009, 0.004])
    measure(ctx, dev, fn, 4, 0.005)
    dev = FakeDevice()
    measure(ctx, dev, scripted(dev, [0.010, 0.002, 0.007]), 2, 0.0045)


@case("stale work and a slow first call, with two samples")
def stale_and_two(ctx):
    dev = FakeDevice()
    fn = scripted(dev, [0.040, 0.004, 0.006])
    dev.launch(0.250)
    measure(ctx, dev, fn, 2, 0.005)
    # After the measurement nothing is left running: the last stop was synchronized.
    check.that(
        dev.pending_s == 0,
        f"{FN}: device work was still running when the function returned",
        f"{dev.pending_s:g} s of queued work had not been waited for",
    )


@case("each call launches several kernels, and launching takes host time")
def several_kernels(ctx):
    # The device works while the host launches: 3 ms + 2 ms of work, 0.2 ms per launch.
    dev = FakeDevice(launch_s=0.0002)
    first, second = dev.kernel(0.003, first_call_s=0.020), dev.kernel(0.002)

    def fn():
        first()
        second()

    measure(ctx, dev, fn, 3, 0.005)
    # When launching costs more than the work, the call is bound by the host.
    dev = FakeDevice(launch_s=0.002)
    measure(ctx, dev, dev.kernel(0.0005), 3, 0.002)


@case("a single sample, and many samples with one outlier")
def sample_counts(ctx):
    dev = FakeDevice()
    measure(ctx, dev, scripted(dev, [0.001, 0.007]), 1, 0.007)
    dev = FakeDevice()
    durations = [0.002, *[0.004] * 10, 0.300, *[0.004] * 10]
    dev.launch(0.5)
    measure(ctx, dev, scripted(dev, durations), 21, 0.004)


@case("random durations: the median of the per-call durations")
def random_durations(ctx):
    for n in (3, 6, 9, 12):
        ms = [int(v) for v in ctx.rng.integers(1, 400, size=n + 1)]
        durations = [v / 1e4 for v in ms]
        dev = FakeDevice()
        dev.launch(float(ctx.rng.integers(1, 50)) / 1e3)
        measure(ctx, dev, scripted(dev, durations), n, statistics.median(durations[1:]))


@case("a real clock on the CPU: RealDevice and a function that sleeps 3 ms")
def real_clock(ctx):
    seconds = ctx.solution.time_region(lambda: time.sleep(0.003), RealDevice("cpu"), 3)
    check.that(isinstance(seconds, float), f"{FN}: must return a Python float")
    check.that(
        0.003 <= seconds < 0.25,
        f"{FN}: measured {seconds:.6f} s per call for a function that sleeps",
        "the function sleeps 0.003 s; the median must be at least that, and close to it",
    )


# --------------------------------------------------------------------------------------
# GPU case. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: the host time is not shorter than what CUDA events measure", requires="cuda")
def gpu_events(ctx):
    # Each call queues eight 2048 x 2048 matrix products (tens of milliseconds of GPU
    # work, launched in well under one) between two timing events. CUDA events are
    # time stamps taken on the device's own timeline, so they need no host clock.
    device = torch.device("cuda")
    a = torch.randn(2048, 2048, device=device)
    pairs = []

    def fn():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = a
        for _ in range(8):
            x = (x @ a) * 1e-3
        end.record()
        pairs.append((start, end))

    n = 5
    call, counters = counted(fn)
    seconds = ctx.solution.time_region(call, RealDevice(device), n)
    torch.cuda.synchronize(device)
    check.that(isinstance(seconds, float), f"{FN}: must return a Python float")
    check.cost(counters, "fn_calls", exactly=n + 1, unit="calls")
    # elapsed_time is in milliseconds; the first pair is the warm-up call.
    on_device = statistics.median(start.elapsed_time(end) for start, end in pairs[1:]) / 1e3
    ctx.metric("median_seconds_per_call", seconds, "s")
    check.that(
        seconds >= 0.5 * on_device,
        f"{FN}: measured {seconds * 1e3:.3f} ms per call, less than half of what the "
        "device's own timing events report for the same calls",
        f"CUDA events: {on_device * 1e3:.3f} ms per call (median). A host timer that is "
        "stopped without torch.cuda.synchronize() measures only the launch.",
    )
