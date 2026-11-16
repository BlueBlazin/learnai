"""Tests for flops-per-token.

The worked examples are compared with literal counts; everything else with the
oracle. All counts are exact integers.
"""

from learnai.judge import case, check

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
}
# https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json: 16 * 128 is not 1024.
QWEN3_06B = {
    "vocab_size": 151936,
    "d_model": 1024,
    "n_layers": 28,
    "n_heads": 16,
    "n_kv_heads": 8,
    "head_dim": 128,
    "mlp_type": "swiglu",
    "d_ff": 3072,
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
}


def flops(ctx, name, cfg, n, label):
    got = ctx.call(name, dict(cfg), n)
    check.that(type(got) is int, f"{label}: the result must be a Python int", f"got: {got!r}")
    return got


def check_literal(ctx, name, cfg, n, expected, label):
    check.equal(flops(ctx, name, cfg, n, label), expected, label=label)


def check_oracle(ctx, name, cfg, n, label):
    check.equal(flops(ctx, name, cfg, n, label), ctx.expected(name, dict(cfg), n), label=label)


def check_both(ctx, cfg, n, description):
    check_oracle(ctx, "decode_flops", cfg, n, f"decode_flops({description}, {n})")
    check_oracle(ctx, "prefill_flops", cfg, n, f"prefill_flops({description}, {n})")


@case("the worked example: Granite 4.2 3B, decode and prefill at 4096 tokens", public=True)
def worked_example(ctx):
    # Rejects decode_quadratic, head_every_position, scores_kv_heads, and the rest: every
    # term of the formula is visible in these two numbers.
    check_literal(
        ctx, "decode_flops", GRANITE_3B, 4096, 8482979840, "decode_flops(granite_3b, 4096)"
    )
    check_literal(
        ctx, "prefill_flops", GRANITE_3B, 4096, 32642265251840, "prefill_flops(granite_3b, 4096)"
    )


@case("the tiny example: two blocks, a two-matrix MLP, one KV head for two heads", public=True)
def tiny_example(ctx):
    check_literal(ctx, "decode_flops", TINY, 5, 864, "decode_flops(tiny, 5)")
    check_literal(ctx, "prefill_flops", TINY, 5, 4000, "prefill_flops(tiny, 5)")
    check_literal(ctx, "prefill_flops", TINY, 1, 672, "prefill_flops(tiny, 1)")
    check_literal(ctx, "decode_flops", TINY, 1, 672, "decode_flops(tiny, 1)")


@case("decode at growing contexts: 1, 2, 1000, 131072", public=True)
def decode_contexts(ctx):
    # Rejects decode_quadratic and decode_excludes_current at sizes that show the growth.
    for context in (1, 2, 1000, 131072):
        check_oracle(
            ctx, "decode_flops", GRANITE_3B, context, f"decode_flops(granite_3b, {context})"
        )
        check_oracle(
            ctx, "decode_flops", GRANITE_350M, context, f"decode_flops(granite_350m, {context})"
        )


@case("prefill at growing lengths: 1, 2, 1000, 32768", public=True)
def prefill_lengths(ctx):
    # Rejects head_every_position on the prefill side.
    for t in (1, 2, 1000, 32768):
        check_oracle(ctx, "prefill_flops", GRANITE_3B, t, f"prefill_flops(granite_3b, {t})")
        check_oracle(ctx, "prefill_flops", GRANITE_350M, t, f"prefill_flops(granite_350m, {t})")


@case("n_heads * head_dim differs from d_model (the shape of Qwen3-0.6B)")
def head_dim_not_d_model(ctx):
    for n in (1, 512, 40960):
        check_both(ctx, QWEN3_06B, n, "qwen3_06b")
    narrow = {**GRANITE_350M, "n_heads": 8, "n_kv_heads": 2, "head_dim": 96}
    check_both(ctx, narrow, 300, "8 heads of width 96 in d_model 1024")


@case("the same model with 16, 8, 2, and 1 KV heads")
def kv_head_counts(ctx):
    # Separates scores_kv_heads from kv_projections_full_width: with n_kv_heads =
    # n_heads both are right, and they diverge in opposite directions below it.
    for n_kv_heads in (16, 8, 2, 1):
        cfg = {**GRANITE_350M, "n_kv_heads": n_kv_heads}
        check_both(ctx, cfg, 2048, f"granite_350m with {n_kv_heads} KV heads")


@case("a two-matrix MLP against a gated MLP of the same width")
def mlp_types(ctx):
    for mlp_type in ("gelu", "swiglu"):
        cfg = {**GRANITE_350M, "mlp_type": mlp_type, "d_ff": 4096}
        check_both(ctx, cfg, 700, f"granite_350m with mlp_type={mlp_type!r}")


@case("extra keys in cfg are ignored, and cfg is not modified")
def full_config(ctx):
    extras = {"attn_bias": True, "mlp_bias": True, "norm_bias": True}
    for tied in (True, False):
        cfg = {**GRANITE_3B, **extras, "tie_embeddings": tied}
        check_both(ctx, cfg, 777, f"granite_3b with biases and tie_embeddings={tied}")
    kept = dict(GRANITE_350M)
    ctx.solution.prefill_flops(kept, 10)
    ctx.solution.decode_flops(kept, 10)
    check.that(kept == GRANITE_350M, "the functions must not modify cfg")


@case("very long prompts: counts beyond 2**53 stay exact")
def beyond_float_precision(ctx):
    for n in (1_000_003, 1_999_999):
        check_both(ctx, GRANITE_3B, n, "granite_3b")
    odd = {**GRANITE_350M, "d_model": 1023, "d_ff": 2047, "vocab_size": 100353, "head_dim": 63}
    check_both(ctx, odd, 1_234_567, "a configuration with odd widths")


@case("random configurations and lengths")
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
        }
        n = int(10 ** rng.uniform(0, 5.5))
        check_both(ctx, cfg, n, f"random configuration {i}")
