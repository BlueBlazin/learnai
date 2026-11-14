"""Tests for real-model-block.

Every expected hidden state is the output of a `transformers` decoder layer, stored
by `fixtures/make.py`; the reference solution is never the oracle.

- The cases marked `assets=[]` run a three-layer Granite with random weights (width
  256: 4 query heads on 2 KV heads, fused SwiGLU 192). Its weights are drawn here from
  the same seed as in `make.py`, so the fixture `tiny.npz` only stores inputs and
  outputs. All of these cases use one input shape, (2, 6, 256): JAX compiles per
  shape.
- The other cases run layers 0, 1, and 13 of the real granite-4.0-350m on hidden
  states of the prompt "The whetstone sharpens the blade, and the blade" (12 tokens).
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import RealModel

FN = "granite_block"
ASSET = "granite-4.0-350m"
REVISION = "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2"
TOL = {"rtol": 1e-3, "atol": 1e-4}  # the tolerance stated in Constraints

WORKED = [-1.174166, -1.392112, -0.481597, 1.608363]  # y[0, 5, :4] of the worked example

TINY = {"d_model": 256, "n_heads": 4, "n_kv_heads": 2, "d_ff": 192, "n_layers": 3}
_cache: dict[str, dict] = {}


def tiny_weights(seed: int = 180) -> dict[str, np.ndarray]:
    """Random weights with Granite 4.0's names and layout. Keep in step with make.py."""
    rng = np.random.default_rng(seed)
    d, f = TINY["d_model"], TINY["d_ff"]

    def normal(rows: int, cols: int, std: float) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * std).astype(np.float32)

    weights = {}
    for i in range(TINY["n_layers"]):
        p = f"model.layers.{i}."
        weights[p + "input_layernorm.weight"] = rng.uniform(0.5, 1.5, d).astype(np.float32)
        weights[p + "self_attn.q_proj.weight"] = normal(TINY["n_heads"] * 64, d, 0.2)
        weights[p + "self_attn.k_proj.weight"] = normal(TINY["n_kv_heads"] * 64, d, 0.2)
        weights[p + "self_attn.v_proj.weight"] = normal(TINY["n_kv_heads"] * 64, d, 0.06)
        weights[p + "self_attn.o_proj.weight"] = normal(d, TINY["n_heads"] * 64, 0.06)
        weights[p + "post_attention_layernorm.weight"] = rng.uniform(0.5, 1.5, d).astype(np.float32)
        weights[p + "shared_mlp.input_linear.weight"] = normal(2 * f, d, 0.08)
        weights[p + "shared_mlp.output_linear.weight"] = normal(d, f, 0.08)
    return weights


def tiny(ctx):
    """(weights, fixture) of the three-layer random model."""
    if "tiny" not in _cache:
        weights = tiny_weights()
        total = float(sum(np.abs(w.astype(np.float64)).sum() for w in weights.values()))
        fixture = ctx.fixture("tiny.npz")
        check.that(
            abs(total - float(fixture["checksum"])) < 1e-6 * total,
            "test setup: the random weights differ from the ones the fixture was made with "
            "(numpy changed its generator?); rerun fixtures/make.py",
        )
        _cache["tiny"] = {"weights": weights, "fixture": fixture}
    return _cache["tiny"]["weights"], _cache["tiny"]["fixture"]


def run(ctx, x, weights, layer, positions, want, label, columns=None):
    got = ctx.call(FN, x, weights, layer, np.asarray(positions))
    check.shape(got, x.shape, label=label)
    if columns is not None:
        got = got[..., columns]
    check.allclose(got, want, **TOL, label=label, dtype=True)
    return got


# ---------------------------------------------------------------------------------
# A three-layer random Granite: no download needed.
# ---------------------------------------------------------------------------------


@case("the worked example: layer 1 of the small random model, two sequences of 6 tokens",
      public=True, assets=[])  # fmt: skip
def worked_example(ctx):
    weights, fixture = tiny(ctx)
    got = run(ctx, fixture["x"], weights, 1, np.arange(6), fixture["y_1"],
              "granite_block(x, weights, 1, [0..5])")  # fmt: skip
    # The numbers printed in the statement: the first four features of the last token
    # of the first sequence.
    check.allclose(got[0, 5, :4], WORKED, **TOL,
                   label="granite_block(x, weights, 1, [0..5])[0, 5, :4]")  # fmt: skip


# Rejects sqrt_scale, no_residual_multiplier, multiplier_on_attention_only,
# multiplier_on_stream, gate_up_swapped, rope_adjacent_pairs, rope_clockwise,
# rope_theta_1e4, kv_heads_tiled, values_rotated, and no_causal_mask.
@case("layers 0 and 2 of the random model",
      public=True, assets=[])  # fmt: skip
def tiny_layers(ctx):
    weights, fixture = tiny(ctx)
    for layer in (0, 2):
        run(ctx, fixture["x"], weights, layer, np.arange(6), fixture[f"y_{layer}"],
            f"granite_block(x, weights, {layer}, [0..5])")  # fmt: skip


# Rejects positions_ignored. Shifting every position by 100 changes nothing (RoPE scores
# depend on differences), which is why the first list has gaps.
@case("positions with gaps [0, 1, 2, 3, 40, 1000], and positions starting at 100",
      public=True, assets=[])  # fmt: skip
def tiny_positions(ctx):
    weights, fixture = tiny(ctx)
    run(ctx, fixture["x"], weights, 0, fixture["spread"], fixture["y_spread"],
        "granite_block(x, weights, 0, [0, 1, 2, 3, 40, 1000])")  # fmt: skip
    run(ctx, fixture["x"], weights, 2, 100 + np.arange(6), fixture["y_offset"],
        "granite_block(x, weights, 2, [100..105])")  # fmt: skip


# Rejects no_causal_mask without any reference.
@case("causality: changing the input from token t on leaves the outputs before t unchanged",
      assets=[])  # fmt: skip
def causality(ctx):
    weights, fixture = tiny(ctx)

    def prop(rng):
        x = fixture["x"].copy()
        t = int(rng.integers(1, 6))
        layer = int(rng.integers(0, 3))
        changed = x.copy()
        changed[:, t:] += rng.standard_normal((2, 6 - t, 256)).astype(np.float32)
        before = ctx.call(FN, x, weights, layer, np.arange(6))
        after = ctx.call(FN, changed, weights, layer, np.arange(6))
        check.shape(before, x.shape, label=FN)
        check.allclose(after[:, :t], before[:, :t], rtol=1e-4, atol=1e-5,
                       label="outputs before the first changed token")  # fmt: skip
        moved = np.abs(after[:, t:] - before[:, t:]).max(axis=-1)
        check.that(
            bool(np.all(moved > 1e-3)),
            "the outputs at the changed tokens did not change",
            f"largest change per row and token: {moved}",
        )

    check.forall(prop, trials=3)


@case("rows of a batch are independent, and other tensors in `weights` are ignored", assets=[])
def batch_rows(ctx):
    weights, fixture = tiny(ctx)
    extra = dict(weights)
    extra["model.embed_tokens.weight"] = ctx.randn(8, 256)
    extra["model.norm.weight"] = ctx.randn(256)
    x = fixture["x"]
    swapped = np.ascontiguousarray(x[::-1])
    run(ctx, swapped, extra, 1, np.arange(6), fixture["y_1"][::-1],
        "granite_block on the two sequences in the other order")  # fmt: skip


# ---------------------------------------------------------------------------------
# granite-4.0-350m: 28 layers, width 1024, 16 query heads on 4 KV heads, SwiGLU 2048.
# ---------------------------------------------------------------------------------


def real(ctx):
    """(weights, fixture) of the real model. Weights are decoded when first touched."""
    fixture = ctx.fixture("granite-4.0-350m.npz")
    check.that(str(fixture["revision"]) == REVISION, "the fixture was made from another revision")
    model = RealModel(ctx.asset(ASSET))
    return model, fixture


@case("Granite 4.0 350M, layer 0: from the scaled embeddings of a 12-token prompt", public=True)
def real_layer_0(ctx):
    model, fixture = real(ctx)
    x = model.lm().embed(fixture["ids"])[None]  # embedding rows times 12: (1, 12, 1024)
    run(ctx, x, model.weights(), 0, np.arange(12), fixture["y_0"],
        "granite_block(x, weights, 0, [0..11]) at 64 columns", fixture["columns"])  # fmt: skip


@case("Granite 4.0 350M, layer 13: a hidden state with entries beyond 10,000", public=True)
def real_layer_13(ctx):
    model, fixture = real(ctx)
    run(ctx, fixture["x_13"], model.weights(), 13, np.arange(12), fixture["y_13"],
        "granite_block(x, weights, 13, [0..11]) at 64 columns", fixture["columns"])  # fmt: skip


# Rejects layer_prefix_without_dot: only a model with more than ten layers has a
# "model.layers.1" that is a prefix of other layers' names.
@case("Granite 4.0 350M, layer 1")
def real_layer_1(ctx):
    model, fixture = real(ctx)
    run(ctx, fixture["x_1"], model.weights(), 1, np.arange(12), fixture["y_1"],
        "granite_block for layer 1 at 64 columns", fixture["columns"])  # fmt: skip


@case("Granite 4.0 350M, layer 13 at positions with gaps")
def real_gapped_positions(ctx):
    model, fixture = real(ctx)
    run(ctx, fixture["x_13"], model.weights(), 13, fixture["gapped"], fixture["y_13_gapped"],
        "granite_block for layer 13 at 64 columns", fixture["columns"])  # fmt: skip
