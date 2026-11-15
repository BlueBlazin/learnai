"""Tests for cuda-memory-accounting.

Everything here is arithmetic on numbers that a GPU reports, so the CPU cases can
check all of it exactly:

- `predicted_allocated` against the allocator's rounding rule (a multiple of 512
  bytes per allocation), computed here independently;
- `fragmentation` and `kv_budget` on dicts with the keys of `torch.cuda.memory_stats()`.
  The dicts are built by `make_stats` below from the documented key names
  (`"<stat>.<pool>.<metric>"`); they were NOT captured from a GPU, because the machine
  that authored the problem has none. The curriculum asked for captured fixtures.
- `measure_peak` against `FakeCuda`, a simulated allocator with the functions of the
  same names in `torch.cuda` (`reset_peak_memory_stats`, `memory_allocated`,
  `max_memory_allocated`, and the two `reserved` ones a wrong solution might call).

Verified on torch 2.15 (CPU build): `torch.cuda.memory_stats()` is an empty dict
without CUDA, and `memory_allocated`, `max_memory_allocated`, `memory_reserved`, and
`max_memory_reserved` are implemented as lookups of `allocated_bytes.all.current`,
`allocated_bytes.all.peak`, `reserved_bytes.all.current`, and `reserved_bytes.all.peak`
in that dict (torch/cuda/memory.py).

The GPU cases (`requires="cuda"`) were written from the PyTorch 2.15 documentation and
have never run on the machine that validates the catalog.
"""

import torch

from whetstone.judge import case, check

KIB, MIB, GIB = 2**10, 2**20, 2**30
BLOCK = 512


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def is_int(value):
    return type(value) is int


def rounded(nbytes):
    return (nbytes + BLOCK - 1) // BLOCK * BLOCK


def blocks(shapes, itemsize):
    """The rule of the statement, written as a loop over elements counts."""
    total = 0
    for shape in shapes:
        count = 1
        for dim in shape:
            count *= dim
        total += rounded(count * itemsize)
    return total


def check_predicted(ctx, shapes, dtype):
    got = ctx.solution.predicted_allocated(list(shapes), dtype)
    check.that(
        is_int(got), f"predicted_allocated: returned a {type(got).__name__}, not a Python int"
    )
    check.equal(got, blocks(shapes, dtype.itemsize), label="predicted_allocated")
    return got


def make_stats(*, small, large, peak, reserved_peak):
    """A dict with the keys of torch.cuda.memory_stats().

    small, large: (allocated, reserved, inactive_split) bytes now, per pool.
    peak, reserved_peak: the high-water marks of allocated and reserved bytes over
    both pools. Counts and historical totals are filled with consistent values.
    """
    pools = {"small_pool": small, "large_pool": large}
    pools["all"] = tuple(a + b for a, b in zip(small, large, strict=True))
    stats = {}
    for pool, (allocated, reserved, split) in pools.items():
        share = allocated / max(1, pools["all"][0])
        byte_stats = {
            "allocated_bytes": (allocated, max(allocated, int(peak * share))),
            "reserved_bytes": (reserved, max(reserved, int(reserved_peak * share))),
            "active_bytes": (allocated, max(allocated, int(peak * share))),
            "requested_bytes": (allocated - allocated // 50, max(allocated, int(peak * share))),
            "inactive_split_bytes": (split, split + split // 3),
        }
        if pool == "all":
            byte_stats["allocated_bytes"] = (allocated, peak)
            byte_stats["active_bytes"] = (allocated, peak)
            byte_stats["reserved_bytes"] = (reserved, reserved_peak)
        for stat, (current, high) in byte_stats.items():
            stats[f"{stat}.{pool}.current"] = current
            stats[f"{stat}.{pool}.peak"] = high
            stats[f"{stat}.{pool}.allocated"] = 3 * high + current
            stats[f"{stat}.{pool}.freed"] = 3 * high
        unit = 4 * KIB if pool == "small_pool" else 8 * MIB
        for stat, current in (
            ("allocated", allocated // unit + 1),
            ("active", allocated // unit + 1),
            ("segment", reserved // (2 * MIB if pool == "small_pool" else 20 * MIB) + 1),
            ("inactive_split", split // unit + 1),
        ):
            stats[f"{stat}.{pool}.current"] = current
            stats[f"{stat}.{pool}.peak"] = 2 * current
            stats[f"{stat}.{pool}.allocated"] = 40 * current
            stats[f"{stat}.{pool}.freed"] = 39 * current
    stats.update(
        num_alloc_retries=0,
        num_ooms=0,
        num_sync_all_streams=2,
        num_device_alloc=17,
        num_device_free=3,
        max_split_size=-1,
    )
    return stats


def check_fragmentation(ctx, stats):
    """fragmentation(stats) against the three definitions, computed from the totals."""
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    split = stats.get("inactive_split_bytes.all.current", 0)
    before = dict(stats)
    got = ctx.solution.fragmentation(stats)
    check.that(stats == before, "fragmentation: modified the stats dict")
    check.that(isinstance(got, dict), f"fragmentation: returned a {type(got).__name__}, not a dict")
    want = {
        "cached_bytes": reserved - allocated,
        "releasable_bytes": reserved - allocated - split,
        "non_releasable_fraction": split / reserved if reserved else 0.0,
    }
    check.that(
        set(got) == set(want),
        "fragmentation: got a dict whose keys are not the expected keys",
        f"expected {sorted(want)}, got {sorted(got)}",
    )
    for key in ("cached_bytes", "releasable_bytes"):
        check.that(
            is_int(got[key]),
            f"fragmentation: {key} is a {type(got[key]).__name__}, not a Python int",
        )
        check.equal(got[key], want[key], label=key)
    check.that(
        isinstance(got["non_releasable_fraction"], float),
        "fragmentation: non_releasable_fraction must be a Python float",
    )
    check.allclose(
        got["non_releasable_fraction"],
        want["non_releasable_fraction"],
        rtol=1e-9,
        atol=1e-12,
        label="non_releasable_fraction",
    )
    return got


def check_budget(ctx, stats, total_bytes, safety):
    """kv_budget against exact integer arithmetic; `safety` is a multiple of 1/64."""
    sixty_fourths = round(safety * 64)
    if sixty_fourths / 64 != safety:
        raise RuntimeError("authoring error: safety must be a multiple of 1/64")
    free = total_bytes - stats.get("allocated_bytes.all.peak", 0)
    want = max(0, free * (64 - sixty_fourths) // 64)
    got = ctx.solution.kv_budget(stats, total_bytes, safety)
    check.that(is_int(got), f"kv_budget: returned a {type(got).__name__}, not a Python int")
    check.equal(got, want, label="kv_budget")
    return got


class FakeCuda:
    """A simulated caching allocator with the memory functions of `torch.cuda`.

    `alloc(nbytes)` and `free(handle)` are what a test's `fn` uses in place of creating
    and deleting CUDA tensors. Allocations are rounded up to 512 bytes; reserved memory
    grows in 2 MiB segments and is never given back, as a cache would behave.
    """

    SEGMENT = 2 * MIB

    def __init__(self):
        self.allocated = self.peak = self.reserved = self.reserved_peak = 0
        self.resets = 0

    def alloc(self, nbytes):
        size = rounded(nbytes)
        self.allocated += size
        self.peak = max(self.peak, self.allocated)
        self.reserved = max(self.reserved, -(-self.allocated // self.SEGMENT) * self.SEGMENT)
        self.reserved_peak = max(self.reserved_peak, self.reserved)
        return size

    def free(self, handle):
        self.allocated -= handle

    def reset_peak_memory_stats(self, device=None):
        self.resets += 1
        self.peak = self.allocated
        self.reserved_peak = self.reserved

    def memory_allocated(self, device=None):
        return self.allocated

    def max_memory_allocated(self, device=None):
        return self.peak

    def memory_reserved(self, device=None):
        return self.reserved

    def max_memory_reserved(self, device=None):
        return self.reserved_peak


def check_peak(ctx, cuda, script, want):
    """measure_peak over a scripted fn: (+n) allocates n bytes, (-k) frees allocation k."""
    calls = []

    def fn():
        calls.append(True)
        handles = []
        for step in script:
            if step > 0:
                handles.append(cuda.alloc(step))
            else:
                cuda.free(handles[-step - 1])
        return handles

    got = ctx.solution.measure_peak(fn, cuda)
    check.that(len(calls) == 1, f"measure_peak: called fn {len(calls)} times, not exactly once")
    check.that(is_int(got), f"measure_peak: returned a {type(got).__name__}, not a Python int")
    check.equal(got, want, label="measure_peak")
    return got


def warmed_up(history_peak, held):
    """An allocator that once held `history_peak` bytes and now holds `held`."""
    cuda = FakeCuda()
    spike = cuda.alloc(history_peak)
    cuda.free(spike)
    cuda.alloc(held)
    return cuda


EXAMPLE_STATS = {
    "small": (16 * MIB, 32 * MIB, 8 * MIB),
    "large": (7 * GIB - 16 * MIB, 10 * GIB - 32 * MIB, GIB // 2 - 8 * MIB),
    "peak": 9 * GIB,
    "reserved_peak": 10 * GIB,
}


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked examples of the statement, one per function", public=True)
def worked_example(ctx):
    got = check_predicted(ctx, [(1,), (1000,), (2, 3, 4096)], torch.float16)
    stats = make_stats(**EXAMPLE_STATS)
    frag = check_fragmentation(ctx, stats)
    budget = check_budget(ctx, stats, 24 * GIB, 0.125)
    cuda = warmed_up(400 * MIB, 100 * MIB)
    peak = check_peak(ctx, cuda, [64 * MIB, 32 * MIB, -1], 96 * MIB)
    if (got, budget, peak) != (51712, 14092861440, 100663296) or frag != {
        "cached_bytes": 3221225472,
        "releasable_bytes": 2684354560,
        "non_releasable_fraction": 0.05,
    }:
        raise RuntimeError("authoring error: the worked example changed")


@case("predicted_allocated: scalars, empty tensors, exact multiples, other dtypes", public=True)
def prediction_rules(ctx):
    # Rejects no_block_rounding, rounds_the_total, empty_takes_a_block, and
    # four_bytes_per_element.
    check_predicted(ctx, [(), (1,), (3,)], torch.float32)  # three blocks of 512
    check_predicted(ctx, [(0,), (4, 0, 8), (128,)], torch.float32)  # two take nothing
    check_predicted(ctx, [(128,), (256,), (2, 64)], torch.float32)  # exact: 512, 1024, 512
    check_predicted(ctx, [(513,), (1, 8, 4096, 64)], torch.bfloat16)
    check_predicted(ctx, [(7, 33), (100,)], torch.int64)
    check_predicted(ctx, [(600,), (5, 5)], torch.bool)
    check_predicted(ctx, [], torch.float32)


@case("fragmentation and kv_budget after a warm-up on a 16 GiB device", public=True)
def after_warmup(ctx):
    # Rejects reserved_as_used, current_not_peak, safety_on_total, split_is_releasable,
    # and large_pool_only.
    stats = make_stats(
        small=(40 * MIB, 64 * MIB, 20 * MIB),
        large=(5 * GIB, 8 * GIB, 300 * MIB),
        peak=6 * GIB + 512 * MIB,
        reserved_peak=8 * GIB + 64 * MIB,
    )
    check_fragmentation(ctx, stats)
    for safety in (0.0, 0.0625, 0.25, 0.5):
        check_budget(ctx, stats, 16 * GIB, safety)


@case("measure_peak: temporaries count, earlier peaks and held memory do not", public=True)
def peak_rules(ctx):
    # Rejects forgets_reset (300 MiB of history), net_growth (the temporary is freed),
    # absolute_peak (100 MiB were held before), and peak_of_reserved.
    check_peak(ctx, warmed_up(300 * MIB, 100 * MIB), [48 * MIB, -1], 48 * MIB)
    # Sizes are rounded up to 512 on the way in: 1024 + 5632 + 512.
    check_peak(ctx, warmed_up(10 * MIB, 10 * MIB), [KIB, 5 * KIB + 1, 3], 7 * KIB)
    # The peak is in the middle of the function, not at its end.
    check_peak(ctx, FakeCuda(), [3 * MIB + 100, 2 * MIB, -1, 3 * MIB], 5 * MIB + 512)


@case("the stats of a machine without CUDA: an empty dict")
def empty_stats(ctx):
    # Rejects requires_keys. torch.cuda.memory_stats() is {} on a CPU-only build.
    got = check_fragmentation(ctx, {})
    check.that(
        got["non_releasable_fraction"] == 0.0,
        "fragmentation: the fraction of nothing reserved must be 0.0",
    )
    check_budget(ctx, {}, 8 * GIB, 0.25)
    check_fragmentation(ctx, torch.cuda.memory_stats() if not torch.cuda.is_available() else {})


@case("kv_budget: a peak above the total gives 0; no safety margin; a full margin of 63/64")
def budget_edges(ctx):
    # Rejects negative_budget.
    stats = make_stats(
        small=(MIB, 2 * MIB, 0), large=(3 * GIB, 4 * GIB, 0), peak=5 * GIB, reserved_peak=6 * GIB
    )
    check_budget(ctx, stats, 4 * GIB, 0.25)
    check_budget(ctx, stats, 5 * GIB, 0.5)
    check_budget(ctx, stats, 5 * GIB + 1000, 0.0)
    check_budget(ctx, stats, 80 * GIB, 0.984375)
    check_budget(ctx, stats, 80 * GIB + 12345, 0.015625)


@case("fragmentation: nothing cached, everything releasable, everything split")
def fragmentation_edges(ctx):
    tight = make_stats(
        small=(2 * MIB, 2 * MIB, 0), large=(GIB, GIB, 0), peak=GIB + 2 * MIB, reserved_peak=2 * GIB
    )
    check_fragmentation(ctx, tight)
    idle = make_stats(
        small=(0, 6 * MIB, 0), large=(0, 3 * GIB, 0), peak=2 * GIB, reserved_peak=4 * GIB
    )
    check_fragmentation(ctx, idle)
    shredded = make_stats(
        small=(MIB, 4 * MIB, 3 * MIB),
        large=(GIB, 3 * GIB, 2 * GIB),
        peak=2 * GIB,
        reserved_peak=3 * GIB + 4 * MIB,
    )
    check_fragmentation(ctx, shredded)


@case("predicted_allocated: a KV cache layer by layer, and a hundred one-element tensors")
def prediction_sizes(ctx):
    check_predicted(ctx, [(4, 8, 2048, 64)] * 80, torch.float16)
    got = check_predicted(ctx, [(1,)] * 100, torch.float32)
    check.that(got == 51200, "predicted_allocated: authoring check failed")
    check_predicted(ctx, [(3, 5, 7), (11, 13), (17,), (2, 2, 2, 2, 3)], torch.float16)


@case("measure_peak twice on one allocator, and a function that only frees")
def peak_sequences(ctx):
    cuda = warmed_up(50 * MIB, 20 * MIB)
    check_peak(ctx, cuda, [200 * MIB, -1], 200 * MIB)
    # The second measurement must not remember the first one's 200 MiB.
    check_peak(ctx, cuda, [10 * MIB, 5 * MIB, -1, -2], 15 * MIB)
    check_peak(ctx, cuda, [], 0)
    check_peak(ctx, cuda, [KIB, -1], 1024)


@case("random shapes, stats, and allocation scripts")
def random_sweep(ctx):
    dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.int64, torch.int8]
    for trial in range(8):
        rng = ctx.child_rng(trial)
        shapes = [
            tuple(int(d) for d in rng.integers(0, 40, size=int(rng.integers(0, 4))))
            for _ in range(int(rng.integers(1, 7)))
        ]
        check_predicted(ctx, shapes, dtypes[trial % len(dtypes)])

        allocated = [int(rng.integers(0, 2**34)) * BLOCK for _ in range(2)]
        cached = [int(rng.integers(0, 2**22)) * BLOCK for _ in range(2)]
        split = [int(rng.integers(0, c // BLOCK + 1)) * BLOCK for c in cached]
        peak = sum(allocated) + int(rng.integers(0, 2**22)) * BLOCK
        stats = make_stats(
            small=(allocated[0] // 4096, allocated[0] // 4096 + cached[0], split[0]),
            large=(allocated[1], allocated[1] + cached[1], split[1]),
            peak=peak,
            reserved_peak=peak + sum(cached),
        )
        check_fragmentation(ctx, stats)
        total = peak + int(rng.integers(-(2**30), 2**36))
        check_budget(ctx, stats, total, float(rng.choice([0.0, 0.03125, 0.125, 0.375, 0.75])))

        sizes = [int(n) for n in rng.integers(1, 5 * MIB, size=int(rng.integers(1, 6)))]
        script, live, current, high = [], [], 0, 0
        for index, size in enumerate(sizes):
            script.append(size)
            live.append(index)
            current += rounded(size)
            high = max(high, current)
            if rng.random() < 0.5:
                victim = live.pop(int(rng.integers(0, len(live))))
                script.append(-(victim + 1))
                current -= rounded(sizes[victim])
        cuda = warmed_up(int(rng.integers(1, 64)) * MIB, int(rng.integers(0, 8)) * MIB)
        check_peak(ctx, cuda, script, high)


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: the prediction is within 2 percent of memory_allocated", requires="cuda")
def gpu_prediction(ctx):
    # Tensors below 1 MiB each, which the allocator serves from its small pool.
    device = torch.device("cuda")
    shapes = [(1,), (3,), (1000,), (7, 129), (64, 64), (255, 513), (2, 3, 4096)]
    for dtype in (torch.float32, torch.float16):
        predicted = ctx.solution.predicted_allocated(shapes, dtype)
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_allocated(device)
        tensors = [torch.empty(shape, dtype=dtype, device=device) for shape in shapes]
        measured = torch.cuda.memory_allocated(device) - before
        del tensors
        ctx.metric(f"measured_bytes_{str(dtype).removeprefix('torch.')}", measured, "bytes")
        check.that(
            abs(predicted - measured) <= 0.02 * measured,
            f"predicted_allocated: {predicted} bytes is not within 2 percent of what the "
            "allocator reports",
            f"torch.cuda.memory_allocated() grew by {measured} bytes",
        )


@case("on the GPU: measure_peak with torch.cuda sees a 64 MiB temporary", requires="cuda")
def gpu_measure_peak(ctx):
    device = torch.device("cuda")
    held = torch.empty(8 * MIB, dtype=torch.uint8, device=device)
    spike = torch.empty(256 * MIB, dtype=torch.uint8, device=device)  # an earlier, larger peak
    del spike

    def fn():
        temporary = torch.empty(64 * MIB, dtype=torch.uint8, device=device)
        del temporary

    got = ctx.solution.measure_peak(fn, torch.cuda)
    del held
    check.that(is_int(got), f"measure_peak: returned a {type(got).__name__}, not a Python int")
    ctx.metric("peak_bytes", got, "bytes")
    check.that(
        64 * MIB <= got <= 66 * MIB,
        f"measure_peak: reported {got} bytes for a function whose only allocation is one "
        "temporary tensor",
        "the temporary is 64 MiB; 8 MiB were held before, and 256 MiB had been allocated "
        "and freed earlier",
    )


@case(
    "on the GPU: fragmentation and kv_budget agree with torch.cuda's own counters", requires="cuda"
)
def gpu_stats(ctx):
    device = torch.device("cuda")
    keep = [torch.empty(n * MIB, dtype=torch.uint8, device=device) for n in (3, 5, 9)]
    del keep[1]
    torch.cuda.synchronize(device)
    stats = torch.cuda.memory_stats(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    peak = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.mem_get_info(device)[1]

    got = ctx.solution.fragmentation(stats)
    check.equal(got["cached_bytes"], reserved - allocated, label="cached_bytes")
    check.that(
        0 <= got["releasable_bytes"] <= got["cached_bytes"],
        "fragmentation: releasable_bytes is not between 0 and cached_bytes",
    )
    check.that(
        0.0 <= got["non_releasable_fraction"] <= 1.0,
        "fragmentation: non_releasable_fraction is not between 0 and 1",
    )
    budget = ctx.solution.kv_budget(stats, total, 0.25)
    check.equal(budget, max(0, (total - peak) * 3 // 4), label="kv_budget")
    del keep
