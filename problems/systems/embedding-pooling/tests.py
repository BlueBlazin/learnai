"""Tests for embedding-pooling.

Numeric cases compare with the oracle on random states whose padded positions hold
large garbage, so a pooling that reads them is wrong by a visible amount. The
property case encodes the same texts with the kit's `TinyEncoder` under different
paddings and requires the same embeddings.

Shapes are shared across cases for JAX's sake: (4, 8, 32), (4, 12, 32), (5, 5, 5),
and the worked example.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import TinyEncoder

MODES = ("mean", "cls", "last")


def padded_batch(rng, lengths, total, dim, side, *, garbage=30.0):
    """Random states for rows with `lengths` real tokens, padded to `total` on `side`;
    the padded positions hold large random values."""
    batch = len(lengths)
    mask = np.zeros((batch, total), dtype=np.int64)
    for row, n in enumerate(lengths):
        if side == "right":
            mask[row, :n] = 1
        else:
            mask[row, total - n :] = 1
    states = rng.standard_normal((batch, total, dim)).astype(np.float32)
    junk = (garbage * rng.standard_normal((batch, total, dim))).astype(np.float32)
    states = np.where(mask[..., None] == 1, states, junk)
    return states, mask


def check_modes(ctx, states, mask, modes=MODES):
    for mode in modes:
        ctx.check("pool", states, mask, mode)


EXAMPLE_STATES = np.array([[[1, 0], [3, 4], [9, 0]], [[0, 2], [4, 0], [2, 1]]], dtype=np.float32)
EXAMPLE_MASK = np.array([[1, 1, 0], [1, 1, 1]])


@case("the worked example: mean, cls, and last on two rows; a truncation", public=True)
def worked_example(ctx):
    r = 0.5**0.5
    want = {
        "mean": [[r, r], [0.8944272, 0.4472136]],
        "cls": [[1.0, 0.0], [0.0, 1.0]],
        "last": [[0.6, 0.8], [0.8944272, 0.4472136]],
    }
    for mode, expected in want.items():
        got = ctx.check("pool", EXAMPLE_STATES, EXAMPLE_MASK, mode)
        check.allclose(got, expected, label=f"pool(states, mask, {mode!r})")
    left = np.array([[[9, 0], [1, 0], [3, 4]]], dtype=np.float32)
    left_mask = np.array([[0, 1, 1]])
    check.allclose(ctx.check("pool", left, left_mask, "cls"), [[1.0, 0.0]], label="cls, left")
    check.allclose(ctx.check("pool", left, left_mask, "last"), [[0.6, 0.8]], label="last, left")
    e = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    got = ctx.check("truncate_matryoshka", e, 2)
    check.allclose(got, [r, r], label="truncate_matryoshka([0.5, 0.5, 0.5, 0.5], 2)")


# Rejects mean_over_padding and pool_not_normalized.
@case("mean pooling ignores the states at padded positions (right and left padding)",
      public=True)  # fmt: skip
def mean_ignores_padding(ctx):
    for side in ("right", "left"):
        states, mask = padded_batch(ctx.rng, [3, 8, 1, 5], 8, 32, side)
        check_modes(ctx, states, mask, ("mean",))


# Rejects last_is_final_column (right padding) and last_assumes_right_padding (left).
@case("last-token pooling finds the last real token under right and under left padding",
      public=True)  # fmt: skip
def last_token_both_sides(ctx):
    for side in ("right", "left"):
        states, mask = padded_batch(ctx.rng, [3, 8, 1, 5], 8, 32, side)
        check_modes(ctx, states, mask, ("last",))


# Rejects cls_is_column_zero.
@case("cls pooling takes the first real token, wherever the padding is", public=True)
def cls_both_sides(ctx):
    for side in ("left", "right"):
        states, mask = padded_batch(ctx.rng, [3, 8, 1, 5], 8, 32, side)
        check_modes(ctx, states, mask, ("cls",))


# Rejects truncate_without_renormalizing.
@case("truncate_matryoshka: prefixes of unit vectors are renormalized", public=True)
def matryoshka(ctx):
    e = ctx.randn(4, 32)
    e /= np.linalg.norm(e, axis=-1, keepdims=True)
    for dim in (8, 16, 32, 1):
        got = ctx.check("truncate_matryoshka", e, dim)
        norms = np.linalg.norm(got, axis=-1)
        check.allclose(norms, np.ones_like(norms), label=f"norms after truncating to {dim}")
    ctx.check("truncate_matryoshka", e[0], 8)
    ctx.check("truncate_matryoshka", 7.5 * e, 16)


# Rejects zero_vector_nan.
@case("a pooled vector that is exactly zero stays zero")
def zero_vector(ctx):
    states, mask = padded_batch(ctx.rng, [2, 8, 4, 6], 8, 32, "right")
    states[0, 1] = -states[0, 0]  # the two real states of row 0 cancel
    states[2, :4] = 0.0  # row 2: every real state is zero
    states[3, 0] = 0.0  # row 3: a zero first token
    states[3, 5] = 0.0  # ... and a zero last token
    for mode in MODES:
        got = ctx.check("pool", states, mask, mode)
        check.finite(got, label=f"pool(states, mask, {mode!r})")
    e = ctx.randn(4, 32)
    e[1, :8] = 0.0
    e[2] = 0.0
    for dim in (8, 16):
        check.finite(ctx.check("truncate_matryoshka", e, dim), label="truncate_matryoshka")


# Rejects normalizes_over_batch: B = D, so a norm over the wrong axis has the right shape.
@case("a square batch (B = T = D = 5), no padding, and full rows in a padded batch")
def square_and_full(ctx):
    states, mask = padded_batch(ctx.rng, [5, 5, 5, 5, 5], 5, 5, "right")
    check_modes(ctx, states, mask)
    states, mask = padded_batch(ctx.rng, [5, 2, 4, 1, 3], 5, 5, "left")
    check_modes(ctx, states, mask)
    e = ctx.randn(5, 5)
    ctx.check("truncate_matryoshka", e, 3)
    ctx.check("truncate_matryoshka", e, 5)


@case("every output has unit length, whatever the scale of the states")
def unit_length(ctx):
    def prop(rng):
        side = ("right", "left")[int(rng.integers(2))]
        lengths = [int(n) for n in rng.integers(1, 9, size=4)]
        states, mask = padded_batch(rng, lengths, 8, 32, side)
        states *= np.float32(10.0 ** rng.uniform(-3, 3))
        for mode in MODES:
            got = ctx.call("pool", states, mask, mode)
            check.shape(got, (4, 32), label=f"pool(..., {mode!r})")
            norms = np.linalg.norm(got, axis=-1)
            check.allclose(norms, np.ones_like(norms), label=f"norms of pool(..., {mode!r})")

    check.forall(prop, trials=6)


def encode(encoder, texts, total, side):
    """Token states and mask of `texts` padded to `total` on `side`."""
    tokens = np.zeros((len(texts), total), dtype=np.int64)
    mask = np.zeros((len(texts), total), dtype=np.int64)
    for row, text in enumerate(texts):
        where = slice(0, len(text)) if side == "right" else slice(total - len(text), total)
        tokens[row, where] = text
        mask[row, where] = 1
    return encoder.reference(tokens, mask), mask


# The property of the statement. Rejects every padding mutant on its own terms: the
# same texts under three paddings must give the same embeddings.
@case("padding never changes the embedding: left, right, and wider padding agree")
def padding_invariance(ctx):
    def prop(rng):
        encoder = TinyEncoder(ctx.fw, seed=int(rng.integers(1, 1000)))
        lengths = [int(n) for n in rng.integers(1, 9, size=4)]
        lengths[int(rng.integers(4))] = 8  # one row fills the narrow batch
        texts = [[1, *(int(t) for t in rng.integers(2, 64, size=n - 1))] for n in lengths]
        layouts = {
            "right padding to 8": encode(encoder, texts, 8, "right"),
            "left padding to 8": encode(encoder, texts, 8, "left"),
            "right padding to 12": encode(encoder, texts, 12, "right"),
            "left padding to 12": encode(encoder, texts, 12, "left"),
        }
        for mode in MODES:
            results = {name: ctx.call("pool", *batch, mode) for name, batch in layouts.items()}
            base = results["right padding to 8"]
            for name, got in results.items():
                check.allclose(
                    got, base, rtol=0, atol=1e-4,
                    label=f"pool(..., {mode!r}) with {name}, against right padding to 8",
                )  # fmt: skip

    check.forall(prop, trials=5)


@case("random batches: both padding sides, every mode, every row length")
def random_batches(ctx):
    rng = ctx.rng
    for trial in range(6):
        side = ("right", "left")[trial % 2]
        total = (8, 12)[trial % 2]
        lengths = [int(n) for n in rng.integers(1, total + 1, size=4)]
        states, mask = padded_batch(rng, lengths, total, 32, side)
        check_modes(ctx, states, mask)
    e = ctx.randn(4, 32, scale=3.0)
    for dim in (8, 16):
        ctx.check("truncate_matryoshka", e, dim)
