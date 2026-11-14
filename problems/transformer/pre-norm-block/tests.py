"""Tests for pre-norm-block.

Every case compares the solution with the oracle. Two of them also pin the wiring down
without it: sublayers whose output projection is zero must leave the stream untouched,
and the block must agree with the kit's `decoder_block_ref` on a one-head config.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import DecoderConfig, decoder_block_ref, decoder_params, layer_params

FN = "block"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def causal(t):
    """True where the key position is not after the query position."""
    return np.tril(np.ones((t, t), dtype=bool))


def config(d_model=8, d_mlp=12):
    # One query head and one key/value head: the single-head attention of this problem.
    return DecoderConfig(d_model=d_model, n_heads=1, n_kv_heads=1, d_mlp=d_mlp, n_layers=1)


def block_params(seed, d_model=8, d_mlp=12):
    """Seeded weights for one block. The two norm gains are different random vectors."""
    return layer_params(decoder_params(config(d_model, d_mlp), seed), 0)


@case("the worked example: two positions of width 2 under a causal mask", public=True)
def worked_example(ctx):
    x = np.array([[[3.0, 4.0], [1.0, -1.0]]], dtype=np.float32)
    params = {
        "input_layernorm.weight": np.array([1.0, 1.0], dtype=np.float32),
        "self_attn.q_proj.weight": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "self_attn.k_proj.weight": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "self_attn.v_proj.weight": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        "self_attn.o_proj.weight": np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        "post_attention_layernorm.weight": np.array([0.5, 2.0], dtype=np.float32),
        "mlp.gate_proj.weight": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "mlp.up_proj.weight": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32),
        "mlp.down_proj.weight": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]], dtype=np.float32),
    }
    got = ctx.check(FN, x, params, causal(2), **TOL)
    expected = np.array([[[1.612561, 13.325579], [-6.983558, 14.508905]]], dtype=np.float32)
    check.allclose(got, expected, rtol=1e-4, atol=1e-5, label=FN)


@case("a random block under a causal mask: B = 2, T = 5, D = 8", public=True)
def random_block(ctx):
    ctx.check(FN, ctx.randn(2, 5, 8), block_params(1), causal(5), **TOL)


# The residual stream is only ever added to. With both output projections at zero the
# sublayers add nothing, so the block is the identity. Rejects post_norm and
# normalizes_residual_stream without reference to the oracle.
@case("sublayers that output zero leave the stream untouched: the result is x", public=True)
def zero_sublayers_are_identity(ctx):
    params = block_params(2)
    params["self_attn.o_proj.weight"] = np.zeros_like(params["self_attn.o_proj.weight"])
    params["mlp.down_proj.weight"] = np.zeros_like(params["mlp.down_proj.weight"])
    x = ctx.randn(2, 5, 8, scale=3.0)
    got = ctx.check(FN, x, params, causal(5), **TOL)
    check.allclose(got, x, rtol=1e-6, atol=1e-6, label="block with zero o_proj and down_proj")


# With the MLP silenced the result is x + attention; with attention silenced it is
# x + mlp(norm2(x)). Each isolates one residual connection.
@case("one sublayer silenced at a time", public=True)
def one_sublayer_at_a_time(ctx):
    x = ctx.randn(2, 5, 8)
    no_mlp = block_params(3)
    no_mlp["mlp.down_proj.weight"] = np.zeros_like(no_mlp["mlp.down_proj.weight"])
    ctx.check(FN, x, no_mlp, causal(5), **TOL)
    no_attn = block_params(3)
    no_attn["self_attn.o_proj.weight"] = np.zeros_like(no_attn["self_attn.o_proj.weight"])
    ctx.check(FN, x, no_attn, causal(5), **TOL)


# The same computation as one block of the kit's decoder with a single head. RoPE at
# position 0 is the identity, so the kit block with all positions 0 has no rotation.
@case("agrees with the kit's decoder block (one head, no rotation)")
def agrees_with_kit_block(ctx):
    x = ctx.randn(2, 5, 8)
    params = block_params(4)
    got = ctx.call(FN, x, params, causal(5))
    expected = decoder_block_ref(config(), params, x, causal(5), positions=np.zeros(5))
    check.allclose(got, expected, **TOL, label=FN)


# Rejects ignores_mask on a mask that is not triangular (a sliding window), and on a
# mask that lets every position see everything (where ignoring it is harmless).
@case("other masks: a window of two positions, and a mask that hides nothing")
def other_masks(ctx):
    x = ctx.randn(2, 5, 8)
    params = block_params(5)
    i, j = np.arange(5)[:, None], np.arange(5)[None, :]
    ctx.check(FN, x, params, (j <= i) & (i - j < 2), **TOL)
    ctx.check(FN, x, params, np.ones((5, 5), dtype=bool), **TOL)


# A stream far from unit scale. The norms see it at unit scale; the residual does not.
# Wirings that put a norm on the stream are off by the factor of the scale here.
@case("a stream with large and with small magnitude")
def stream_scales(ctx):
    params = block_params(6)
    ctx.check(FN, ctx.randn(2, 5, 8, scale=40.0), params, causal(5), **TOL)
    ctx.check(FN, ctx.randn(2, 5, 8, scale=0.05), params, causal(5), **TOL)


# Gains that are far apart: sharing one norm, or swapping the two, cannot hide.
@case("norm gains that differ strongly between the two sublayers")
def distinct_gains(ctx):
    params = block_params(7)
    params["input_layernorm.weight"] = ctx.uniform(8, low=0.2, high=0.6)
    params["post_attention_layernorm.weight"] = ctx.uniform(8, low=1.5, high=3.0)
    ctx.check(FN, ctx.randn(2, 5, 8), params, causal(5), **TOL)


@case("a single position: T = 1")
def single_position(ctx):
    ctx.check(FN, ctx.randn(2, 1, 8), block_params(8), causal(1), **TOL)


# B = T = D = M: every axis has the same length, which hides transposed weights.
@case("all axes the same length: B = T = D = M = 4")
def all_axes_equal(ctx):
    ctx.check(FN, ctx.randn(4, 4, 4), block_params(9, d_model=4, d_mlp=4), causal(4), **TOL)


@case("a wider block: D = 16 with M = 40")
def wider_block(ctx):
    ctx.check(FN, ctx.randn(2, 5, 16), block_params(10, d_model=16, d_mlp=40), causal(5), **TOL)
