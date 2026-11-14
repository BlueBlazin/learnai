"""Tests for memory-bandwidth-bound.

The worked example is compared with literal values; everything else with the oracle.
Rates are floats compared to a relative tolerance of 1e-9; crossover batches are exact,
and random inputs are kept away from the knife edges the statement excludes.
"""

import math

from whetstone.judge import case, check

GB = 10**9
# lmstudio-community/granite-4.2-3b-MLX-8bit, model.safetensors (bytes on the Hub).
W_8BIT = 3_888_771_048
# ibm-granite/granite-4.2-3b, both BF16 shards.
W_BF16 = 5_358_533_712 + 1_960_983_408
# ibm-granite/granite-4.0-350m, BF16.
W_350M = 704_786_224
# One sequence's KV cache at 4096 tokens: 81920 bytes per token (Granite 4.2 3B, BF16).
K_4096 = 81_920 * 4096
RTOL = 1e-9


def rate(ctx, weight_bytes, kv_bytes_read, bandwidth, batch, label):
    got = ctx.call("decode_tokens_per_s", weight_bytes, kv_bytes_read, bandwidth, batch)
    check.that(
        isinstance(got, float), f"{label}: the result must be a Python float", f"got: {got!r}"
    )
    return got


def check_rate(ctx, weight_bytes, kv_bytes_read, bandwidth, batch, label=None, expected=None):
    label = label or f"decode_tokens_per_s({weight_bytes}, {kv_bytes_read}, {bandwidth}, {batch})"
    got = rate(ctx, weight_bytes, kv_bytes_read, bandwidth, batch, label)
    if expected is None:
        expected = ctx.expected(
            "decode_tokens_per_s", weight_bytes, kv_bytes_read, bandwidth, batch
        )
    check.allclose(got, expected, rtol=RTOL, atol=0.0, label=label)


def check_crossover(ctx, weight_bytes, kv_bytes_read, bandwidth, compute, label=None, **kwargs):
    label = label or f"crossover_batch({weight_bytes}, {kv_bytes_read}, {bandwidth}, {compute})"
    assert not on_knife_edge(weight_bytes, kv_bytes_read, bandwidth, compute), label
    got = ctx.call("crossover_batch", weight_bytes, kv_bytes_read, bandwidth, compute)
    check.that(
        got is None or type(got) is int,
        f"{label}: the result must be a Python int or None",
        f"got: {got!r}",
    )
    if "expected" in kwargs:
        expected = kwargs["expected"]
    else:
        expected = ctx.expected("crossover_batch", weight_bytes, kv_bytes_read, bandwidth, compute)
    check.that(
        got == expected,
        f"{label}: wrong crossover batch",
        f"expected: {expected!r}\n     got: {got!r}",
    )


def on_knife_edge(weight_bytes, kv_bytes_read, bandwidth, compute):
    """Whether rounding could change the crossover (the statement excludes such inputs)."""
    spare = bandwidth * GB - compute * kv_bytes_read
    if abs(spare) <= 1e-5 * bandwidth * GB:
        return True
    if spare < 0:
        return False
    threshold = compute * weight_bytes / spare
    return threshold > 100_000 or abs(threshold - round(threshold)) <= 1e-5 * max(1.0, threshold)


@case("the worked example: 8-bit Granite 4.2 3B on 400 GB/s", public=True)
def worked_example(ctx):
    # Rejects gib_not_gb outright; weights_per_sequence and per_stream_rate at batch 8;
    # ignores_kv wherever the cache is read.
    check_rate(ctx, W_8BIT, 0, 400.0, 1, "decode_tokens_per_s(W, 0, 400.0, 1)", 102.86025972285525)
    check_rate(
        ctx, W_8BIT, K_4096, 400.0, 1, "decode_tokens_per_s(W, K, 400.0, 1)", 94.68990005577633
    )
    check_rate(
        ctx, W_8BIT, K_4096, 400.0, 8, "decode_tokens_per_s(W, K, 400.0, 8)", 486.8308002672813
    )
    check_crossover(
        ctx, W_8BIT, K_4096, 400.0, 800.0, "crossover_batch(W, K, 400.0, 800.0)", expected=24
    )
    check_crossover(ctx, W_8BIT, 0, 400.0, 800.0, "crossover_batch(W, 0, 400.0, 800.0)", expected=8)
    check_crossover(
        ctx, W_8BIT, K_4096, 400.0, 1200.0, "crossover_batch(W, K, 400.0, 1200.0)", expected=None
    )
    check_crossover(
        ctx, W_8BIT, K_4096, 400.0, 90.0, "crossover_batch(W, K, 400.0, 90.0)", expected=1
    )


@case("throughput against batch: 1, 2, 4, 16, 64, 1024 sequences", public=True)
def batch_scaling(ctx):
    # Rejects weights_per_sequence and per_stream_rate.
    for batch in (1, 2, 4, 16, 64, 1024):
        check_rate(ctx, W_8BIT, K_4096, 400.0, batch)
        check_rate(ctx, W_BF16, K_4096 // 4, 120.0, batch)


@case("an empty cache: the bound is batch * bandwidth / weight_bytes", public=True)
def no_cache(ctx):
    for batch in (1, 3, 32):
        check_rate(ctx, W_350M, 0, 100.0, batch)
        check_rate(ctx, W_BF16, 0, 400.0, batch)
    check_crossover(ctx, W_350M, 0, 100.0, 500.0)
    check_crossover(ctx, W_BF16, 0, 400.0, 2000.0)


@case("crossover: compute rates below, between, and above the bound's limits", public=True)
def crossover_regimes(ctx):
    # Rejects crossover_rounds_down (thresholds are far from integers) and
    # no_saturation_check (rates above bandwidth / kv_bytes_read).
    for compute in (50.0, 94.0, 95.0, 300.0, 1000.0, 1150.0, 1190.0, 1195.0, 5000.0):
        check_crossover(ctx, W_8BIT, K_4096, 400.0, compute)


@case("fractional bandwidths and bandwidths below 1 GB/s")
def bandwidth_units(ctx):
    # Rejects gib_not_gb where no other effect is in play.
    for bandwidth in (0.5, 25.6, 68.25, 409.6, 3350.0):
        check_rate(ctx, W_350M, 28_672 * 2048, bandwidth, 1)
        check_rate(ctx, W_350M, 28_672 * 2048, bandwidth, 12)
        check_crossover(ctx, W_350M, 28_672 * 2048, bandwidth, 40.0)


@case("a cache larger than the weights: long contexts on a small model")
def cache_dominates(ctx):
    # Rejects ignores_kv: here the cache, not the model, sets the ceiling.
    kv = 28_672 * 32_768  # Granite 4.0 350M at its full context, BF16: 940 MB
    for batch in (1, 5, 100, 10_000):
        check_rate(ctx, W_350M, kv, 200.0, batch)
    for compute in (100.0, 180.0, 210.0, 212.0, 213.0, 250.0):
        check_crossover(ctx, W_350M, kv, 200.0, compute)


@case("crossover at batch 1, at batch 2, and at a batch in the thousands")
def crossover_extremes(ctx):
    check_crossover(ctx, 10**9, 10**6, 100.0, 1.0)
    check_crossover(ctx, 10**9, 10**6, 100.0, 99.5)
    check_crossover(ctx, 10**9, 10**6, 100.0, 150.0)
    check_crossover(ctx, 10**9, 10**6, 100.0, 9_000.0)
    check_crossover(ctx, 10**9, 10**6, 100.0, 97_000.0)
    check_crossover(ctx, 10**9, 10**6, 100.0, 103_000.0)


@case("the crossover is the first batch whose bound reaches the compute rate")
def crossover_consistent(ctx):
    # Ties the two functions together without the oracle.
    cases = (
        (W_8BIT, K_4096, 400.0, 800.0),
        (W_BF16, K_4096, 400.0, 333.0),
        (W_350M, 0, 50.0, 1234.5),
    )
    for weight_bytes, kv, bandwidth, compute in cases:
        label = f"crossover_batch({weight_bytes}, {kv}, {bandwidth}, {compute})"
        b = ctx.call("crossover_batch", weight_bytes, kv, bandwidth, compute)
        check.that(type(b) is int and b >= 1, f"{label}: expected an int batch >= 1", f"got: {b!r}")
        at = rate(ctx, weight_bytes, kv, bandwidth, b, label)
        check.that(
            at >= compute,
            f"{label}: decode_tokens_per_s at the returned batch is below the compute rate",
            f"returned batch: {b}\nrate there: {at!r}\ncompute rate: {compute!r}",
        )
        if b > 1:
            before = rate(ctx, weight_bytes, kv, bandwidth, b - 1, label)
            check.that(
                before < compute,
                f"{label}: a smaller batch already reaches the compute rate",
                f"returned batch: {b}\nrate at {b - 1}: {before!r}\ncompute rate: {compute!r}",
            )


@case("random sizes, bandwidths, batches, and compute rates")
def random_inputs(ctx):
    rng = ctx.rng
    done = 0
    while done < 60:
        weight_bytes = int(10 ** rng.uniform(6, 11.5))
        kv = 0 if rng.random() < 0.15 else int(10 ** rng.uniform(3, 10))
        bandwidth = round(float(10 ** rng.uniform(0, 3.7)), 2)
        batch = int(10 ** rng.uniform(0, 4))
        check_rate(ctx, weight_bytes, kv, bandwidth, batch)
        # Compute rates spread around the two scales of the bound: B / W and B / K.
        single = bandwidth * GB / weight_bytes
        ceiling = bandwidth * GB / kv if kv else single * 1000
        low, high = min(single, ceiling) / 4, max(single, ceiling) * 2
        compute = float(math.exp(rng.uniform(math.log(low), math.log(high))))
        if not on_knife_edge(weight_bytes, kv, bandwidth, compute):
            check_crossover(ctx, weight_bytes, kv, bandwidth, compute)
            done += 1
