"""Tests for param-count.

The worked examples are compared with literal counts (they are the published parameter
counts of real checkpoints); everything else is compared with the oracle. All counts
are exact integers.
"""

from learnai.judge import case, check

KEYS = ("embedding", "attention", "mlp", "norm", "head", "total")
NO_BIAS = {"attn_bias": False, "mlp_bias": False, "norm_bias": False}

# https://huggingface.co/ibm-granite/granite-4.2-3b/raw/main/config.json
GRANITE_3B = {
    "vocab_size": 100352,
    "d_model": 2560,
    "n_layers": 40,
    "n_heads": 40,
    "n_kv_heads": 8,
    "head_dim": 64,
    "mlp_type": "swiglu",
    "d_ff": 8192,
    "tie_embeddings": False,
    **NO_BIAS,
}
# https://huggingface.co/ibm-granite/granite-4.0-350m/raw/main/config.json
GRANITE_350M = {
    "vocab_size": 100352,
    "d_model": 1024,
    "n_layers": 28,
    "n_heads": 16,
    "n_kv_heads": 4,
    "head_dim": 64,
    "mlp_type": "swiglu",
    "d_ff": 2048,
    "tie_embeddings": True,
    **NO_BIAS,
}
# https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json (QK-norm gains left out)
QWEN3_06B = {
    "vocab_size": 151936,
    "d_model": 1024,
    "n_layers": 28,
    "n_heads": 16,
    "n_kv_heads": 8,
    "head_dim": 128,
    "mlp_type": "swiglu",
    "d_ff": 3072,
    "tie_embeddings": True,
    **NO_BIAS,
}
TINY = {
    "vocab_size": 10,
    "d_model": 4,
    "n_layers": 2,
    "n_heads": 2,
    "n_kv_heads": 1,
    "head_dim": 3,
    "mlp_type": "gelu",
    "d_ff": 8,
    "tie_embeddings": False,
    "attn_bias": True,
    "mlp_bias": True,
    "norm_bias": True,
}


def count(ctx, cfg, label):
    """Call the solution and check the shape of the answer before its numbers."""
    got = ctx.call("count_params", dict(cfg))
    check.that(isinstance(got, dict), f"{label}: count_params must return a dict")
    check.that(
        set(got) == set(KEYS),
        f"{label}: the result must have exactly the keys {', '.join(KEYS)}",
        f"got keys: {sorted(got, key=str)!r}",
    )
    check.that(
        all(type(got[key]) is int for key in KEYS),
        f"{label}: every count must be a Python int",
        f"got: {got!r}",
    )
    return got


def check_literal(ctx, cfg, expected, label):
    got = count(ctx, cfg, label)
    check.equal(got, dict(zip(KEYS, expected, strict=True)), label=label)


def check_oracle(ctx, cfg, label):
    got = count(ctx, cfg, label)
    check.equal(got, ctx.expected("count_params", dict(cfg)), label=label)


@case("the worked examples: Granite 4.2 3B (untied) and Granite 4.0 350M (tied)", public=True)
def worked_example(ctx):
    # The 350M model rejects tied_head_counted_twice; both reject gqa_full_heads and
    # swiglu_two_matrices.
    check_literal(
        ctx,
        GRANITE_3B,
        (256901120, 629145600, 2516582400, 207360, 256901120, 3659737600),
        "count_params(granite_3b)",
    )
    check_literal(
        ctx,
        GRANITE_350M,
        (102760448, 73400320, 176160768, 58368, 0, 352379904),
        "count_params(granite_350m)",
    )


@case("the tiny example: a two-matrix MLP with every bias switched on", public=True)
def tiny_with_biases(ctx):
    # Rejects bias_per_input, and forgets_final_norm on a count small enough to redo by hand.
    check_literal(ctx, TINY, (40, 176, 152, 40, 40, 448), "count_params(tiny)")


@case("n_heads * head_dim differs from d_model (the shape of Qwen3-0.6B)", public=True)
def head_dim_not_d_model(ctx):
    # Rejects assumes_heads_fill_d_model: 16 heads of 128 are 2048 wide, d_model is 1024.
    check_literal(
        ctx,
        QWEN3_06B,
        (155582464, 176160768, 264241152, 58368, 0, 596042752),
        "count_params(qwen3_06b)",
    )
    narrow = {**GRANITE_350M, "n_heads": 8, "n_kv_heads": 2, "head_dim": 96}
    check_oracle(ctx, narrow, "count_params with 8 heads of width 96 in d_model 1024")


@case("tying and untying the head of the same model", public=True)
def tie_toggle(ctx):
    for base, name in ((GRANITE_350M, "granite_350m"), (GRANITE_3B, "granite_3b")):
        for tied in (True, False):
            cfg = {**base, "tie_embeddings": tied}
            check_oracle(ctx, cfg, f"count_params({name}, tie_embeddings={tied})")


@case("each bias flag on its own, for both MLP types")
def bias_flags(ctx):
    # Rejects bias_per_input: widths are chosen so that q, kv, d_model, and d_ff differ.
    base = {**GRANITE_350M, "d_model": 96, "n_heads": 6, "n_kv_heads": 2, "head_dim": 20}
    base |= {"d_ff": 250, "vocab_size": 1000, "n_layers": 3}
    for mlp_type in ("swiglu", "gelu"):
        for flag in ("attn_bias", "mlp_bias", "norm_bias"):
            cfg = {**base, "mlp_type": mlp_type, flag: True}
            check_oracle(ctx, cfg, f"count_params with {mlp_type} and only {flag}")
        cfg = {**base, "mlp_type": mlp_type, "attn_bias": True, "mlp_bias": True}
        check_oracle(ctx, cfg, f"count_params with {mlp_type}, attn_bias and mlp_bias")


@case("multi-head (n_kv_heads = n_heads) and multi-query (n_kv_heads = 1) attention")
def kv_head_extremes(ctx):
    for n_kv_heads in (16, 8, 2, 1):
        cfg = {**GRANITE_350M, "n_kv_heads": n_kv_heads}
        check_oracle(ctx, cfg, f"count_params with 16 heads and {n_kv_heads} KV heads")


@case("a classic two-matrix MLP and a gated MLP of the same width")
def mlp_types(ctx):
    for mlp_type in ("gelu", "swiglu"):
        for d_ff in (4096, 2731):
            cfg = {**GRANITE_350M, "mlp_type": mlp_type, "d_ff": d_ff}
            check_oracle(ctx, cfg, f"count_params with mlp_type={mlp_type!r}, d_ff={d_ff}")


@case("one block, and a model small enough that the final norm is visible")
def single_layer(ctx):
    # Rejects forgets_final_norm with nothing else in the way.
    cfg = {**TINY, "n_layers": 1, "norm_bias": False, "attn_bias": False, "mlp_bias": False}
    check_oracle(ctx, cfg, "count_params with one block")
    check_oracle(ctx, {**cfg, "norm_bias": True}, "count_params with one block and norm_bias")
    check_oracle(ctx, {**GRANITE_3B, "n_layers": 1}, "count_params(granite_3b with one block)")


@case("the configuration is not modified")
def cfg_unchanged(ctx):
    cfg = dict(GRANITE_350M)
    ctx.solution.count_params(cfg)
    check.that(cfg == GRANITE_350M, "count_params must not modify cfg")


@case("random configurations")
def random_configs(ctx):
    rng = ctx.rng
    for i in range(40):
        n_kv_heads = int(rng.integers(1, 9))
        cfg = {
            "vocab_size": int(rng.integers(100, 300_000)),
            "d_model": int(rng.integers(8, 5000)),
            "n_layers": int(rng.integers(1, 100)),
            "n_heads": n_kv_heads * int(rng.integers(1, 9)),
            "n_kv_heads": n_kv_heads,
            "head_dim": int(rng.integers(1, 300)),
            "mlp_type": ("swiglu", "gelu")[int(rng.integers(2))],
            "d_ff": int(rng.integers(8, 20_000)),
            "tie_embeddings": bool(rng.integers(2)),
            "attn_bias": bool(rng.integers(2)),
            "mlp_bias": bool(rng.integers(2)),
            "norm_bias": bool(rng.integers(2)),
        }
        check_oracle(ctx, cfg, f"count_params(random configuration {i})")
