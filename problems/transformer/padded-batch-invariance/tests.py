"""Tests for padded-batch-invariance.

The property under test: row b of the batched result equals the last-token logits of
the kit's oracle, `decoder_forward_ref`, run on seqs[b] alone and without any padding.
Numeric cases check it on chosen batches, the property cases on random length mixes.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import DecoderConfig, decoder_forward_ref, decoder_params

FN = "batched_forward"
TOL = {"rtol": 1e-3, "atol": 1e-4}  # the tolerance stated in Constraints (a deep stack)


def alone(cfg, params, seqs):
    """Each sequence through the oracle by itself: the answer batching must not change."""
    return np.stack([decoder_forward_ref(cfg, params, np.array(seq))[-1] for seq in seqs])


def run(ctx, cfg, seed, seqs, pad_id):
    params = decoder_params(cfg, seed)
    got = ctx.check(FN, params, cfg, seqs, pad_id, **TOL)
    check.allclose(got, alone(cfg, params, seqs), **TOL, label=FN)
    return got


def random_seq(ctx, cfg, length, avoid=None):
    """`length` random token ids, none equal to `avoid`."""
    seq = [int(t) for t in ctx.rng.integers(0, cfg.vocab_size, length)]
    return [t if t != avoid else (t + 1) % cfg.vocab_size for t in seq]


@case("the worked example: lengths 3, 1, and 2 in one batch, pad_id = 0", public=True)
def worked_example(ctx):
    cfg = DecoderConfig(vocab_size=5, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_mlp=6)
    got = run(ctx, cfg, 0, [[1, 3, 4], [2], [4, 0]], 0)
    expected = np.array(
        [[-0.525848, -3.150511, 2.285472, 0.695466, -1.054566],
         [-0.656692, -3.295562, 2.559741, 2.571999, -1.360637],
         [0.285671, -1.890282, -2.674756, -4.402627, -2.909733]],
        dtype=np.float32,
    )  # fmt: skip
    check.allclose(got, expected, **TOL, label=FN)


@case("the default model: lengths 6, 3, and 1", public=True)
def mixed_lengths(ctx):
    cfg = DecoderConfig()
    seqs = [random_seq(ctx, cfg, n, avoid=0) for n in (6, 3, 1)]
    run(ctx, cfg, 1, seqs, 0)


# No padding at all: right_pad_last_column and every mask mutant except
# pad_mask_replaces_causal pass here.
@case("equal lengths: nothing is padded", public=True)
def equal_lengths(ctx):
    cfg = DecoderConfig()
    run(ctx, cfg, 2, [random_seq(ctx, cfg, 6, avoid=0) for _ in range(3)], 0)


@case("random length mixes: every row equals its sequence run alone", public=True)
def random_length_mixes(ctx):
    cfg = DecoderConfig()
    params = decoder_params(cfg, 3)

    def prop(rng):
        lengths = [int(n) for n in rng.integers(1, 7, size=3)]
        lengths[int(rng.integers(3))] = 6  # the longest row is always 6: one tensor shape
        pad_id = int(rng.integers(cfg.vocab_size))
        seqs = [random_seq(ctx, cfg, n, avoid=pad_id) for n in lengths]
        got = ctx.call(FN, params, cfg, seqs, pad_id)
        check.shape(got, (3, cfg.vocab_size), label=FN)
        check.allclose(
            got, alone(cfg, params, seqs), **TOL,
            label=f"{FN} for lengths {lengths} versus each sequence alone",
        )  # fmt: skip

    check.forall(prop, trials=4)


# Rejects mask_from_pad_id_values: real tokens that equal pad_id, in the middle of a
# sequence, at its start, and as its last token.
@case("pad_id also occurs as a real token")
def pad_id_is_a_real_token(ctx):
    cfg = DecoderConfig()
    pad = 9
    seqs = [[4, pad, 17, pad, 30, 2], [pad, pad, 5, 11], [21, pad]]
    run(ctx, cfg, 4, seqs, pad)


# The padding token's embedding must not reach the result. Rejects pads_attended and
# mask_on_query_axis without the oracle.
@case("the result does not depend on which token is used for padding")
def pad_token_is_irrelevant(ctx):
    cfg = DecoderConfig()
    params = decoder_params(cfg, 5)
    seqs = [random_seq(ctx, cfg, n) for n in (2, 6, 4)]
    first = ctx.call(FN, params, cfg, seqs, 0)
    second = ctx.call(FN, params, cfg, seqs, 63)
    check.allclose(first, second, rtol=1e-4, atol=1e-4, label=f"{FN} with pad_id 63 versus pad_id 0")
    check.allclose(first, alone(cfg, params, seqs), **TOL, label=FN)


@case("reordering the sequences reorders the rows and changes nothing else")
def row_order(ctx):
    cfg = DecoderConfig()
    params = decoder_params(cfg, 6)
    seqs = [random_seq(ctx, cfg, n, avoid=1) for n in (1, 6, 3)]
    forward = ctx.call(FN, params, cfg, seqs, 1)
    backward = ctx.call(FN, params, cfg, seqs[::-1], 1)
    check.allclose(backward[::-1], forward, rtol=1e-4, atol=1e-4, label="rows after reordering")
    check.allclose(forward, alone(cfg, params, seqs), **TOL, label=FN)


# Padding rows have no key to attend to. In a deep model whatever they contain has many
# layers in which to leak into the real rows. Rejects nan_from_masked_rows.
@case("a 4-layer model with heavy padding: lengths 1, 6, and 2")
def deep_model_heavy_padding(ctx):
    cfg = DecoderConfig(n_layers=4)
    got = run(ctx, cfg, 7, [random_seq(ctx, cfg, n, avoid=0) for n in (1, 6, 2)], 0)
    check.finite(got, label=FN)


@case("another architecture: an untied head and a single key/value head")
def other_architecture(ctx):
    cfg = DecoderConfig(tie_embeddings=False, n_kv_heads=1, rope_base=100.0)
    run(ctx, cfg, 8, [random_seq(ctx, cfg, n, avoid=0) for n in (5, 6, 2)], 0)
