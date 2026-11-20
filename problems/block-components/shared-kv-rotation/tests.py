"""Tests for shared-kv-rotation.

`relative_form` below is the definition written the other way round, in numpy and
without any absolute position: scores and outputs are built from rotations by the
position *difference* of key and query. Cases compare the solution with the oracle,
with that form, and with outputs of the attention code of `transformers`'
`deepseek_v4` on the model's sizes (`fixtures/deepseek_v4.npz`, `fixtures/make.py`).

Angles are float32 products of positions and frequencies, so cases with positions of
1000 and more use the looser tolerance of the statement.
"""

import numpy as np

from learnai.judge import case, check

FN = "shared_kv_attention"
TOL = {"rtol": 1e-4, "atol": 1e-4}  # positions below 64
FAR = {"rtol": 1e-3, "atol": 2e-3}  # positions below 2048
H, T, D, R = 2, 5, 8, 4  # the sizes most cases share (JAX compiles once per shape)


def causal(t, s=None):
    return np.tril(np.ones((t, s or t), dtype=bool))


def turn(x, angles):
    """Rotate interleaved pairs of `x` (..., 2n) counterclockwise by `angles` (..., n)."""
    a, b = x[..., 0::2], x[..., 1::2]
    cos, sin = np.cos(angles), np.sin(angles)
    return np.stack((a * cos - b * sin, a * sin + b * cos), axis=-1).reshape(x.shape)


def relative_form(q, kv, pos_q, pos_k, mask, rope_dim, base):
    """The output written with relative angles only. float64 numpy; q is (H, T, d)."""
    q, kv = q.astype(np.float64), kv.astype(np.float64)
    d = q.shape[-1]
    split = d - rope_dim
    freqs = base ** (-np.arange(0, rope_dim, 2) / rope_dim)
    # rel[t, s]: the key's rotary slice turned by (pos_k[s] - pos_q[t]) * freqs.
    delta = (pos_k[None, :] - pos_q[:, None])[..., None] * freqs  # (T, S, rope_dim / 2)
    rel = turn(np.broadcast_to(kv[None, :, split:], (*delta.shape[:2], rope_dim)), delta)
    scores = q[..., :split] @ kv[:, :split].T + np.einsum("htc,tsc->hts", q[..., split:], rel)
    scores = np.where(mask, scores / np.sqrt(d), -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    plain = weights @ kv[:, :split]
    rotary = np.einsum("hts,tsc->htc", weights, rel)
    return np.concatenate((plain, rotary), axis=-1).astype(np.float32), weights


def inputs(ctx, h=H, t=T, s=T, d=D, gain=2.0):
    return ctx.randn(h, t, d, scale=gain), ctx.randn(s, d)


@case("the worked example: two tokens at positions 3 and 4, one rotated pair", public=True)
def worked_example(ctx):
    q = np.array([[[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]], dtype=np.float32)
    kv = np.array([[1.0, 2.0, 1.0, 0.0], [2.0, -1.0, 0.0, 1.0]], dtype=np.float32)
    pos = np.array([3, 4])
    got = ctx.check(FN, q, kv, pos, pos, causal(2), 2, 100.0, **TOL)
    expected = [[[1.0, 2.0, 1.0, 0.0], [1.584548, 0.246357, 0.224470, 0.234957]]]
    check.allclose(got, np.array(expected, np.float32), atol=1e-5, label=FN)


@case("causal self-attention: 2 heads, 5 tokens, width 8, the last 4 channels rotate", public=True)
def random_causal(ctx):
    q, kv = inputs(ctx)
    ctx.check(FN, q, kv, np.arange(T), np.arange(T), causal(T), R, 10000.0, **TOL)


# The library's attention on the model's sizes: head width 512, trailing 64 channels
# rotated in interleaved pairs, scale 512^-0.5, the output turned back by the query
# position. A wrong slice, pairing, frequency ladder, or sign fails against it.
@case("DeepSeek-V4 sizes: width 512 with a rotary slice of 64, against transformers", public=True)
def deepseek_v4_sizes(ctx):
    f = ctx.fixture("deepseek_v4.npz")
    rope_dim, base = int(f["rope_dim"]), float(f["base"])
    pos = f["pos_near"]
    got = ctx.call(FN, f["q"], f["kv"], pos, pos, causal(len(pos)), rope_dim, base)
    check.allclose(got, f["out_near"], **TOL, label=f"{FN} versus transformers")


# Not about shift invariance: one visible key, so the weight is 1 and the output is that
# key with its rotary slice turned by the relative angle. A solution that hands back
# unrotated values (each value turned back by its own position) returns the key itself.
@case("one visible key 3 positions back: the output is the key turned by -3 steps", public=True)
def single_key_relative_angle(ctx):
    q, kv = inputs(ctx, t=1, s=1)
    pos_q, pos_k = np.array([10]), np.array([7])
    got = ctx.call(FN, q, kv, pos_q, pos_k, np.ones((1, 1), dtype=bool), R, 100.0)
    check.shape(got, (H, 1, D), label=FN)
    freqs = 100.0 ** (-np.arange(0, R, 2) / R)
    expected = np.concatenate((kv[0, : D - R], turn(kv[0, D - R :], -3 * freqs)))
    check.allclose(
        got,
        np.broadcast_to(expected, got.shape),
        **TOL,
        label="output for a single key at relative position -3",
    )


@case("adding a constant to every position leaves the output unchanged", public=True)
def shift_invariance(ctx):
    def prop(rng):
        q, kv = inputs(ctx)
        pos = np.sort(rng.choice(40, size=T, replace=False))
        shift = int(rng.integers(1, 24))
        here = ctx.call(FN, q, kv, pos, pos, causal(T), R, 1000.0)
        moved = ctx.call(FN, q, kv, pos + shift, pos + shift, causal(T), R, 1000.0)
        check.shape(here, q.shape, label=FN)
        check.allclose(
            moved, here, rtol=1e-3, atol=1e-3, label=f"output after shifting positions by {shift}"
        )

    check.forall(prop, trials=4)


# Rejects query_positions_from_zero, inverse_by_key_positions, and every solution that
# restarts positions per call.
@case("cached decoding: one query at position 37 over six cached rows")
def cached_decode(ctx):
    q, kv = inputs(ctx, t=1, s=6)
    pos_k = 32 + np.arange(6)
    mask = np.ones((1, 6), dtype=bool)
    got = ctx.check(FN, q, kv, np.array([37]), pos_k, mask, R, 10000.0, **TOL)
    expected, _ = relative_form(q, kv, np.array([37]), pos_k, mask, R, 10000.0)
    check.allclose(got, expected, **TOL, label=f"{FN} versus the relative form")


@case("the output equals the form written with position differences only")
def relative_form_matches(ctx):
    def prop(rng):
        q, kv = inputs(ctx)
        pos = np.sort(rng.choice(48, size=T, replace=False))
        got = ctx.call(FN, q, kv, pos, pos, causal(T), R, 1000.0)
        expected, _ = relative_form(q, kv, pos, pos, causal(T), R, 1000.0)
        check.allclose(got, expected, **TOL, label=f"{FN} versus the relative form")

    check.forall(prop, trials=4)


# The channels before the rotary slice are an ordinary weighted average of the rows of
# kv: no rotation may touch them on the way in or on the way out.
@case("the leading channels are plain attention over the unrotated channels")
def leading_channels_plain(ctx):
    q, kv = inputs(ctx)
    pos = np.array([3, 4, 9, 20, 21])
    got = ctx.call(FN, q, kv, pos, pos, causal(T), R, 10000.0)
    _, weights = relative_form(q, kv, pos, pos, causal(T), R, 10000.0)
    check.allclose(
        got[..., : D - R],
        (weights @ kv[:, : D - R].astype(np.float64)).astype(np.float32),
        **TOL,
        label="leading channels of the output",
    )


@case("the whole head rotates (rope_dim = d), and a single rotated pair")
def slice_extremes(ctx):
    q, kv = inputs(ctx)
    pos = np.arange(T)
    for rope_dim in (D, 2):
        ctx.check(FN, q, kv, pos, pos, causal(T), rope_dim, 10000.0, **TOL)


@case("a sliding-window mask, and positions with gaps")
def window_and_gaps(ctx):
    q, kv = inputs(ctx)
    window = causal(T) & ~np.tril(np.ones((T, T), dtype=bool), -3)  # self and two back
    pos = np.array([0, 1, 2, 30, 31])
    ctx.check(FN, q, kv, pos, pos, window, R, 10000.0, **TOL)
    ctx.check(FN, q, kv, pos, pos, causal(T), R, 50.0, **TOL)


# H = T = S: an axis mix-up between heads, queries, and keys keeps every shape.
@case("as many heads as queries as keys, and more keys than queries")
def square_and_rectangular(ctx):
    q, kv = inputs(ctx, h=5, t=5, s=5)
    ctx.check(FN, q, kv, np.arange(5), np.arange(5), causal(5), R, 10000.0, **TOL)
    q, kv = inputs(ctx, t=2, s=6)
    mask = np.array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=bool)
    ctx.check(FN, q, kv, np.array([14, 15]), 10 + np.arange(6), mask, R, 10000.0, **TOL)


# Each query points straight at the row after it, which the causal mask hides: that
# row's score is about 560, far above every visible one. A mask that only subtracts a
# finite amount from the score (30, 100, 500) leaves it the winner. Rejects
# additive_finite_mask.
@case("a hidden row with a far higher score than every visible one")
def masked_row_would_win(ctx):
    kv = np.zeros((T, D), dtype=np.float32)
    kv[np.arange(T), np.arange(T)] = 4.0  # row s is 4 * e_s, in the unrotated channels
    kv[:, D - 2 :] = ctx.randn(T, 2)
    q = ctx.randn(H, T, D)
    q[:, np.arange(T - 1), np.arange(1, T)] += 400.0  # query t points at row t + 1
    pos = np.arange(T)
    got = ctx.check(FN, q, kv, pos, pos, causal(T), 2, 10000.0, **TOL)
    expected, _ = relative_form(q, kv, pos, pos, causal(T), 2, 10000.0)
    check.finite(got, label=FN)
    check.allclose(got, expected, **TOL, label=f"{FN} versus the relative form")


# Confident attention: scores spread over tens of units. Rejects no_scale by a wide
# margin.
@case("sharp attention: large queries")
def sharp_attention(ctx):
    q, kv = inputs(ctx, gain=12.0)
    pos = np.arange(T)
    got = ctx.check(FN, q, kv, pos, pos, causal(T), R, 10000.0, **TOL)
    check.finite(got, label=FN)


@case("the DeepSeek-V4 sizes again, 1500 positions into the context")
def deepseek_v4_far(ctx):
    f = ctx.fixture("deepseek_v4.npz")
    rope_dim, base = int(f["rope_dim"]), float(f["base"])
    pos = f["pos_far"]
    got = ctx.call(FN, f["q"], f["kv"], pos, pos, causal(len(pos)), rope_dim, base)
    check.allclose(got, f["out_far"], **FAR, label=f"{FN} versus transformers")
    check.allclose(got, f["out_near"], **FAR, label=f"{FN} versus the same rows elsewhere")
