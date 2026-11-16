"""Tests for memory-budget-fit.

The worked example is compared with literal counts; everything else with the oracle
and, in the property-style sweep, with the definition itself (the answer fits, one
more does not). All results are exact integers.
"""

from learnai.judge import case, check

DTYPES = ("float32", "float16", "bfloat16", "int8")
ITEMSIZE = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}

GIB = 2**30
# Attention geometry from each model's config.json on the Hugging Face Hub; blocks of
# 16 tokens are vLLM's.
GRANITE_3B = {"n_layers": 40, "n_heads": 40, "n_kv_heads": 8, "head_dim": 64, "block_tokens": 16}
GRANITE_350M = {"n_layers": 28, "n_heads": 16, "n_kv_heads": 4, "head_dim": 64, "block_tokens": 16}
QWEN3_06B = {"n_layers": 28, "n_heads": 16, "n_kv_heads": 8, "head_dim": 128, "block_tokens": 16}
# Weight bytes: parameters * 2 for the BF16 checkpoints, the file's data section for
# the 8-bit build (problems/sizing/real-model-sizing).
GRANITE_3B_BF16 = 7_319_475_200
GRANITE_3B_8BIT = 3_888_665_600
GRANITE_350M_BF16 = 704_759_808


def describe(memory, weights, cfg, dtype, amount, percent):
    return (
        f"memory_bytes = {memory}\nweight_bytes = {weights}\ncfg = {cfg!r}\n"
        f"dtype = {dtype!r}\ncontext or batch = {amount}\noverhead_percent = {percent}\n"
    )


def call_int(ctx, name, args, label):
    memory, weights, cfg, dtype, amount, percent = args
    got = ctx.call(name, memory, weights, dict(cfg), dtype, amount, percent)
    check.that(
        type(got) is int,
        f"{label}: the result must be a Python int",
        describe(*args) + f"got: {got!r}",
    )
    return got


def check_literal(ctx, name, args, expected, label):
    check.equal(call_int(ctx, name, args, label), expected, label=label)


def check_oracle(ctx, name, args, label):
    memory, weights, cfg, dtype, amount, percent = args
    expected = ctx.expected(name, memory, weights, dict(cfg), dtype, amount, percent)
    got = call_int(ctx, name, args, label)
    check.that(
        got == expected,
        f"{label}: got {got}, which is not the expected count",
        describe(*args) + f"expected: {expected}\n     got: {got}",
    )


def both(ctx, memory, weights, cfg, dtype, percent, contexts, batches, label):
    for context in contexts:
        args = (memory, weights, cfg, dtype, context, percent)
        check_oracle(ctx, "max_sequences", args, f"max_sequences, {label}, context {context}")
    for batch in batches:
        args = (memory, weights, cfg, dtype, batch, percent)
        check_oracle(ctx, "max_context", args, f"max_context, {label}, batch {batch}")


@case("the worked example: Granite 4.2 3B in BF16 on a 24 GiB device", public=True)
def worked_example(ctx):
    # Rejects weights_forgotten, overhead_after_division, overhead_on_remainder,
    # ignores_blocks; the 1000-token context rejects partial_block_free and the
    # 131072-token one rounds_up.
    m, w, cfg = 24 * GIB, GRANITE_3B_BF16, GRANITE_3B
    for context, expected in ((4096, 47), (1000, 192), (131072, 1)):
        check_literal(
            ctx, "max_sequences", (m, w, cfg, "bfloat16", context, 10), expected,
            f"max_sequences(24 GiB, granite_3b BF16, context {context}, 10%)",
        )  # fmt: skip
    for batch, expected in ((1, 193760), (8, 24208), (64, 3024)):
        check_literal(
            ctx, "max_context", (m, w, cfg, "bfloat16", batch, 10), expected,
            f"max_context(24 GiB, granite_3b BF16, batch {batch}, 10%)",
        )  # fmt: skip


@case("the same model on 8 GiB and on 8 GB: barely fits, and does not fit", public=True)
def small_devices(ctx):
    # Rejects negative_not_clamped (8 GB: the weights and the reserve exceed the device).
    cfg = GRANITE_3B
    for memory, name, sequences, context in (
        (8 * GIB, "8 GiB", 1, 5008),
        (8 * 10**9, "8 GB", 0, 0),
    ):
        check_literal(
            ctx, "max_sequences", (memory, GRANITE_3B_BF16, cfg, "bfloat16", 4096, 10), sequences,
            f"max_sequences({name}, granite_3b BF16, context 4096, 10%)",
        )  # fmt: skip
        check_literal(
            ctx, "max_context", (memory, GRANITE_3B_BF16, cfg, "bfloat16", 1, 10), context,
            f"max_context({name}, granite_3b BF16, batch 1, 10%)",
        )  # fmt: skip
    # The 8-bit build of the same model leaves room for eleven sequences on 8 GiB.
    check_literal(
        ctx, "max_sequences", (8 * GIB, GRANITE_3B_8BIT, cfg, "bfloat16", 4096, 10), 11,
        "max_sequences(8 GiB, granite_3b 8-bit, context 4096, 10%)",
    )  # fmt: skip


@case("contexts around a block boundary: 1, 15, 16, 17, 4095, 4096, 4097 tokens", public=True)
def block_boundaries(ctx):
    # Rejects partial_block_free and ignores_blocks.
    for context in (1, 15, 16, 17, 4095, 4096, 4097):
        args = (16 * GIB, GRANITE_350M_BF16, GRANITE_350M, "bfloat16", context, 10)
        check_oracle(ctx, "max_sequences", args, f"max_sequences(16 GiB, granite_350m, {context})")


@case("overhead from 0 to 99 percent, and every cache dtype", public=True)
def overheads_and_dtypes(ctx):
    for percent in (0, 1, 10, 25, 50, 99):
        both(
            ctx, 24 * GIB, GRANITE_3B_8BIT, GRANITE_3B, "bfloat16", percent, (2048, 3000), (1, 6),
            f"granite_3b 8-bit on 24 GiB, overhead {percent}%",
        )  # fmt: skip
    for dtype in DTYPES:
        both(
            ctx, 12 * GIB, GRANITE_350M_BF16, QWEN3_06B, dtype, 10, (1000, 32768), (3, 32),
            f"cache dtype {dtype}",
        )  # fmt: skip


@case("the pool is not a multiple of the batch, and a batch larger than the pool")
def uneven_batches(ctx):
    # Rejects rounds_up: leftover blocks cannot be handed out.
    cfg = {"n_layers": 2, "n_heads": 4, "n_kv_heads": 2, "head_dim": 8, "block_tokens": 16}
    block_bytes = 16 * 2 * 2 * 2 * 8 * 2
    memory, weights = 1000 * block_bytes + 5000, 5000
    for batch in (1, 3, 7, 999, 1000, 1001, 4096):
        args = (memory, weights, cfg, "bfloat16", batch, 0)
        check_oracle(ctx, "max_context", args, f"max_context, a pool of 1000 blocks, batch {batch}")
    for context in (16, 48, 49, 15984, 16000, 16001):
        args = (memory, weights, cfg, "bfloat16", context, 0)
        check_oracle(
            ctx, "max_sequences", args, f"max_sequences, a pool of 1000 blocks, context {context}"
        )


@case("the cache budget lands exactly on, one byte below, and one byte above a block")
def exact_fit(ctx):
    # Rejects reserve_rounded_down: with the reserve rounded up the budget is one byte
    # short of the 500th block.
    cfg = {"n_layers": 4, "n_heads": 8, "n_kv_heads": 2, "head_dim": 16, "block_tokens": 32}
    block_bytes = 32 * 2 * 4 * 2 * 16 * 4
    memory = 10**9 + 7  # not a multiple of 100: the 10 percent reserve is fractional
    floor_reserve = memory * 10 // 100
    for delta in (-1, 0, 1, 2):
        weights = memory - floor_reserve - 500 * block_bytes + delta
        both(
            ctx, memory, weights, cfg, "float32", 10, (32, 320, 16000), (1, 2, 250),
            f"weights {delta:+d} bytes from an exact fit of 500 blocks",
        )  # fmt: skip
    # A device whose size is a multiple of 100: the reserve is whole.
    memory = 10**9
    weights = memory - memory // 10 - 500 * block_bytes
    both(ctx, memory, weights, cfg, "float32", 10, (32, 16000), (1, 500), "an exact fit")


@case("nothing fits: weights larger than the device, and less than one block left")
def nothing_fits(ctx):
    # Rejects negative_not_clamped.
    cfg = GRANITE_350M
    for memory, weights, label in (
        (GIB, 2 * GIB, "weights twice the device"),
        (GIB, GIB, "weights equal to the device"),
        (GIB, GIB - GIB // 10 + 5, "the reserve does not fit"),
        (GIB, GIB - (GIB * 10 + 99) // 100 - 100, "100 bytes left for the cache"),
        (16 * 28672 * 100, 16 * 28672 * 90 - 16 * 28672 + 1, "one byte short of a block"),
    ):
        both(ctx, memory, weights, cfg, "bfloat16", 10, (1, 16, 4096), (1, 8), label)


@case("extra keys in cfg are ignored, a block of 1 token, and the arguments are not modified")
def blocks_and_extras(ctx):
    cfg = {**QWEN3_06B, "vocab_size": 151936, "d_model": 1024, "block_tokens": 1}
    both(ctx, 6 * GIB, 1_200_000_000, cfg, "float16", 5, (1, 777, 40960), (1, 13), "block of 1")
    cfg = {**QWEN3_06B, "block_tokens": 256}
    both(ctx, 6 * GIB, 1_200_000_000, cfg, "float16", 5, (1, 257, 40960), (1, 13), "block of 256")
    kept = dict(GRANITE_3B)
    ctx.solution.max_sequences(24 * GIB, GRANITE_3B_BF16, kept, "bfloat16", 4096, 10)
    ctx.solution.max_context(24 * GIB, GRANITE_3B_BF16, kept, "bfloat16", 4, 10)
    check.that(kept == GRANITE_3B, "the functions must not modify cfg")


def fits(memory, weights, cfg, dtype, percent, batch, context):
    """The definition, written out: do `batch` sequences of `context` tokens fit?"""
    reserved = -(-memory * percent // 100)
    token_bytes = 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * ITEMSIZE[dtype]
    blocks = -(-context // cfg["block_tokens"])
    needed = weights + batch * blocks * cfg["block_tokens"] * token_bytes
    return needed <= memory - reserved


@case("random devices and models: the answer fits and one more does not")
def random_budgets(ctx):
    rng = ctx.rng
    for i in range(60):
        cfg = {
            "n_layers": int(rng.integers(1, 60)),
            "n_heads": 16,
            "n_kv_heads": int(rng.integers(1, 17)),
            "head_dim": int(rng.integers(1, 200)),
            "block_tokens": int(rng.choice([1, 8, 16, 32, 100])),
        }
        dtype = DTYPES[int(rng.integers(len(DTYPES)))]
        memory = int(rng.integers(10**8, 10**11))
        weights = int(memory * rng.uniform(0.0, 1.05))
        percent = int(rng.integers(0, 40))
        context = int(rng.integers(1, 50_000))
        batch = int(rng.integers(1, 300))
        base = (memory, weights, cfg, dtype)

        args = (*base, context, percent)
        label = f"max_sequences(random case {i})"
        check_oracle(ctx, "max_sequences", args, label)
        n = ctx.call("max_sequences", *base[:2], dict(cfg), dtype, context, percent)
        check.that(
            n == 0 or fits(*base, percent, n, context),
            f"{label}: that many sequences do not fit in the budget",
            describe(*args) + f"got: {n}",
        )
        check.that(
            not fits(*base, percent, n + 1, context),
            f"{label}: one more sequence would still fit",
            describe(*args) + f"got: {n}",
        )

        args = (*base, batch, percent)
        label = f"max_context(random case {i})"
        check_oracle(ctx, "max_context", args, label)
        c = ctx.call("max_context", *base[:2], dict(cfg), dtype, batch, percent)
        check.that(
            c == 0 or fits(*base, percent, batch, c),
            f"{label}: a context that long does not fit in the budget",
            describe(*args) + f"got: {c}",
        )
        check.that(
            not fits(*base, percent, batch, c + 1),
            f"{label}: a longer context would still fit",
            describe(*args) + f"got: {c}",
        )
