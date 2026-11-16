"""Tests for kv-cache-bytes.

The worked examples are compared with literal byte counts; everything else with the
oracle. All results are exact integers.
"""

from learnai.judge import case, check

DTYPES = ("float32", "float16", "bfloat16", "int8")

# Attention geometry from each model's config.json on the Hugging Face Hub.
GRANITE_3B = {"n_layers": 40, "n_heads": 40, "n_kv_heads": 8, "head_dim": 64}
GRANITE_350M = {"n_layers": 28, "n_heads": 16, "n_kv_heads": 4, "head_dim": 64}
QWEN3_06B = {"n_layers": 28, "n_heads": 16, "n_kv_heads": 8, "head_dim": 128}
GEMMA3_4B = {"n_layers": 34, "n_heads": 8, "n_kv_heads": 4, "head_dim": 256}
# Gemma 3: five sliding-window layers (1024 tokens), then one global layer, repeating.
GEMMA3_WINDOWS = {layer: 1024 for layer in range(34) if (layer + 1) % 6 != 0}


def call_int(ctx, name, *args, label):
    got = ctx.call(name, *args)
    check.that(type(got) is int, f"{label}: the result must be a Python int", f"got: {got!r}")
    return got


def per_token_literal(ctx, cfg, dtype, expected, label):
    got = call_int(ctx, "kv_bytes_per_token", dict(cfg), dtype, label=label)
    check.equal(got, expected, label=label)


def per_token_oracle(ctx, cfg, dtype, label):
    got = call_int(ctx, "kv_bytes_per_token", dict(cfg), dtype, label=label)
    check.equal(got, ctx.expected("kv_bytes_per_token", dict(cfg), dtype), label=label)


def total_literal(ctx, cfg, dtype, batch, context, windows, expected, label):
    got = call_int(ctx, "kv_bytes", dict(cfg), dtype, batch, context, dict(windows), label=label)
    check.equal(got, expected, label=label)


def total_oracle(ctx, cfg, dtype, batch, context, windows, label):
    got = call_int(ctx, "kv_bytes", dict(cfg), dtype, batch, context, dict(windows), label=label)
    expected = ctx.expected("kv_bytes", dict(cfg), dtype, batch, context, dict(windows))
    check.equal(got, expected, label=label)


@case("the worked example: Granite 4.2 3B, 8 KV heads shared by 40 query heads", public=True)
def worked_example(ctx):
    # Rejects query_heads, keys_only, bits_not_bytes, bfloat16_four_bytes; the batch of
    # eight rejects ignores_batch.
    name = "granite_3b"
    per_token_literal(ctx, GRANITE_3B, "bfloat16", 81920, f"kv_bytes_per_token({name}, 'bfloat16')")
    per_token_literal(ctx, GRANITE_3B, "float32", 163840, f"kv_bytes_per_token({name}, 'float32')")
    total_literal(
        ctx, GRANITE_3B, "bfloat16", 1, 131072, {}, 10737418240,
        f"kv_bytes({name}, 'bfloat16', 1, 131072, {{}})",
    )  # fmt: skip
    total_literal(
        ctx, GRANITE_3B, "bfloat16", 8, 4096, {}, 2684354560,
        f"kv_bytes({name}, 'bfloat16', 8, 4096, {{}})",
    )  # fmt: skip


@case("the Gemma 3 4B example: 29 sliding-window layers and 5 global ones", public=True)
def sliding_windows(ctx):
    # Rejects window_ignored (context past the window) and window_on_every_layer.
    name = "gemma3_4b"
    per_token_literal(ctx, GEMMA3_4B, "bfloat16", 139264, f"kv_bytes_per_token({name}, 'bfloat16')")
    for context, windows, expected in (
        (512, GEMMA3_WINDOWS, 71303168),
        (8192, GEMMA3_WINDOWS, 289406976),
        (8192, {}, 1140850688),
    ):
        shown = "windows" if windows else "{}"
        total_literal(
            ctx, GEMMA3_4B, "bfloat16", 1, context, windows, expected,
            f"kv_bytes({name}, 'bfloat16', 1, {context}, {shown})",
        )  # fmt: skip


@case("every dtype: float32, float16, bfloat16, int8", public=True)
def every_dtype(ctx):
    for dtype in DTYPES:
        per_token_oracle(ctx, GRANITE_350M, dtype, f"kv_bytes_per_token(granite_350m, {dtype!r})")
        total_oracle(
            ctx, QWEN3_06B, dtype, 3, 1000, {}, f"kv_bytes(qwen3_06b, {dtype!r}, 3, 1000, {{}})"
        )


@case("batch sizes from 1 to 256", public=True)
def batch_axis(ctx):
    # Rejects ignores_batch.
    for batch in (1, 2, 7, 64, 256):
        total_oracle(
            ctx, GRANITE_350M, "float16", batch, 2048, {},
            f"kv_bytes(granite_350m, 'float16', {batch}, 2048, {{}})",
        )  # fmt: skip
        total_oracle(
            ctx, GEMMA3_4B, "bfloat16", batch, 3000, GEMMA3_WINDOWS,
            f"kv_bytes(gemma3_4b, 'bfloat16', {batch}, 3000, windows)",
        )  # fmt: skip


@case("context at, just below, and just above a window; context 0 and 1")
def window_boundaries(ctx):
    for context in (0, 1, 1023, 1024, 1025, 2048, 131072):
        total_oracle(
            ctx, GEMMA3_4B, "bfloat16", 2, context, GEMMA3_WINDOWS,
            f"kv_bytes(gemma3_4b, 'bfloat16', 2, {context}, windows)",
        )  # fmt: skip
        total_oracle(
            ctx, GRANITE_3B, "bfloat16", 2, context, {},
            f"kv_bytes(granite_3b, 'bfloat16', 2, {context}, {{}})",
        )  # fmt: skip


@case("layers with different windows, a single windowed layer, and all layers windowed")
def mixed_windows(ctx):
    # Rejects window_on_every_layer: the windows differ from layer to layer.
    cfg = {"n_layers": 6, "n_heads": 8, "n_kv_heads": 2, "head_dim": 32}
    window_sets = {
        "windows of 16, 64, and 256 on layers 0, 2, 5": {0: 16, 2: 64, 5: 256},
        "a window of 100 on layer 3 only": {3: 100},
        "a window of 128 on every layer": dict.fromkeys(range(6), 128),
        "a window of 1 on layer 0": {0: 1},
    }
    for description, windows in window_sets.items():
        for context in (10, 100, 5000):
            total_oracle(
                ctx, cfg, "float32", 3, context, windows,
                f"kv_bytes at context {context} with {description}",
            )  # fmt: skip


@case("multi-head attention (as many KV heads as heads) and a single KV head")
def kv_head_extremes(ctx):
    for n_kv_heads in (16, 4, 1):
        cfg = {"n_layers": 12, "n_heads": 16, "n_kv_heads": n_kv_heads, "head_dim": 48}
        label = f"with 16 heads and {n_kv_heads} KV heads"
        per_token_oracle(ctx, cfg, "float16", f"kv_bytes_per_token {label}")
        total_oracle(ctx, cfg, "float16", 5, 777, {2: 100}, f"kv_bytes {label}")


@case("extra keys in cfg are ignored, and the arguments are not modified")
def full_config(ctx):
    cfg = {
        **QWEN3_06B,
        "vocab_size": 151936,
        "d_model": 1024,
        "d_ff": 3072,
        "mlp_type": "swiglu",
        "tie_embeddings": True,
    }
    per_token_literal(ctx, cfg, "bfloat16", 114688, "kv_bytes_per_token(qwen3_06b, 'bfloat16')")
    total_oracle(
        ctx, cfg, "bfloat16", 4, 40960, {}, "kv_bytes(qwen3_06b, 'bfloat16', 4, 40960, {})"
    )
    kept_cfg, kept_windows = dict(GEMMA3_4B), dict(GEMMA3_WINDOWS)
    ctx.solution.kv_bytes(kept_cfg, "int8", 2, 4096, kept_windows)
    check.that(
        kept_cfg == GEMMA3_4B and kept_windows == GEMMA3_WINDOWS,
        "kv_bytes must not modify cfg or window_layers",
    )


@case("random configurations, dtypes, batches, contexts, and windows")
def random_configs(ctx):
    rng = ctx.rng
    for i in range(40):
        n_layers = int(rng.integers(1, 60))
        n_kv_heads = int(rng.integers(1, 17))
        cfg = {
            "n_layers": n_layers,
            "n_heads": n_kv_heads * int(rng.integers(1, 6)),
            "n_kv_heads": n_kv_heads,
            "head_dim": int(rng.integers(1, 300)),
        }
        dtype = DTYPES[int(rng.integers(len(DTYPES)))]
        windows = {
            layer: int(2 ** rng.integers(0, 14)) for layer in range(n_layers) if rng.random() < 0.4
        }
        batch = int(rng.integers(1, 100))
        context = int(rng.integers(0, 20_000))
        per_token_oracle(ctx, cfg, dtype, f"kv_bytes_per_token(random configuration {i})")
        total_oracle(ctx, cfg, dtype, batch, context, windows, f"kv_bytes(random case {i})")
