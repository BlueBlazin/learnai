"""Tests for partial-rope.

Numeric cases compare with the oracle; two cases compare with arrays computed by the
rotary code of `transformers` for the RoPE parameters of Qwen3.5 and Gemma 4
(`fixtures/shipped.npz`, written by `fixtures/make.py`). Angles are positions times
frequencies in float32, so the error grows with the position: positions below 64 use a
tight tolerance, larger ones a looser one (see the statement's Constraints).
"""

import numpy as np

from learnai.judge import case, check

PARTIAL, PROPORTIONAL = "partial_rope", "proportional_rope"
BOTH = (PARTIAL, PROPORTIONAL)
TIGHT = {"rtol": 1e-4, "atol": 1e-4}  # positions below 64
LOOSE = {"rtol": 1e-3, "atol": 5e-3}  # positions below 2048
SHAPE = (2, 3, 5, 16)  # the shape most cases share (JAX compiles once per shape)


def pos(*values):
    return np.array(values, dtype=np.int64)


def unrotated(fn, dim, factor):
    """The channels `fn` must return unchanged, as a boolean mask over the head."""
    keep = np.ones(dim, dtype=bool)
    if fn == PARTIAL:
        keep[: int(dim * factor)] = False
    else:
        n = int(factor * dim // 2)
        keep[:n] = False
        keep[dim // 2 : dim // 2 + n] = False
    return keep


@case("the worked example: eight ones at position 1, base = 100, factor = 0.5", public=True)
def worked_example(ctx):
    x = np.ones((1, 1, 1, 8), dtype=np.float32)
    got = ctx.check(PARTIAL, x, pos(1), 100.0, 0.5, **TIGHT)
    expected = [-0.301169, 0.895171, 1.381773, 1.094838, 1.0, 1.0, 1.0, 1.0]
    check.allclose(got.reshape(-1), np.array(expected, np.float32), atol=1e-5, label=PARTIAL)
    got = ctx.check(PROPORTIONAL, x, pos(1), 100.0, 0.5, **TIGHT)
    expected = [-0.301169, 0.639432, 1.0, 1.0, 1.381773, 1.261399, 1.0, 1.0]
    check.allclose(got.reshape(-1), np.array(expected, np.float32), atol=1e-5, label=PROPORTIONAL)


@case("a random tensor of shape (2, 3, 5, 16), positions 0 to 4, factor = 0.5", public=True)
def random_tensor(ctx):
    for fn in BOTH:
        ctx.check(fn, ctx.randn(*SHAPE), np.arange(5), 10000.0, 0.5, **TIGHT)


# The two conventions on the numbers of the models that ship them. A solution that
# swaps the frequency denominators, the pairing, or the rotated block fails here against
# the library, whatever this problem's reference does.
@case("Qwen3.5 (default, 0.25 of 256) and Gemma 4 (proportional, 0.25 of 512)", public=True)
def shipped_parameters(ctx):
    f = ctx.fixture("shipped.npz")
    got = ctx.call(PARTIAL, f["qwen_x"], f["qwen_pos_near"], float(f["qwen_theta"]), 0.25)
    check.allclose(got, f["qwen_y_near"], **TIGHT, label="partial_rope, Qwen3.5 parameters")
    got = ctx.call(PROPORTIONAL, f["gemma_x"], f["gemma_pos_near"], float(f["gemma_theta"]), 0.25)
    check.allclose(got, f["gemma_y_near"], **TIGHT, label="proportional_rope, Gemma 4 parameters")


@case("channels outside the rotated set come back unchanged", public=True)
def unrotated_channels(ctx):
    x = ctx.randn(*SHAPE)
    for fn in BOTH:
        for factor in (0.25, 0.5, 0.75):
            got = ctx.call(fn, x, pos(3, 11, 12, 20, 63), 10000.0, factor)
            check.shape(got, x.shape, label=fn)
            keep = unrotated(fn, 16, factor)
            check.allclose(
                got[..., keep],
                x[..., keep],
                label=f"{fn} with factor {factor}: the channels that must not rotate",
            )
            moved = np.abs(got - x).max(axis=(0, 1, 2)) > 1e-3
            check.that(
                bool(moved[~keep].all()),
                f"{fn} with factor {factor}: a channel that must rotate came back unchanged",
                f"channels that changed: {np.flatnonzero(moved).tolist()}\n"
                f"channels that must change: {np.flatnonzero(~keep).tolist()}",
            )


@case("attention scores depend only on the difference of positions", public=True)
def relative_position(ctx):
    def prop(rng):
        q, k = ctx.randn(*SHAPE), ctx.randn(*SHAPE)
        m, n = rng.integers(0, 40, size=5), rng.integers(0, 40, size=5)
        shift = int(rng.integers(1, 24))
        factor = float(rng.choice([0.25, 0.5, 0.75]))
        for fn in BOTH:

            def scores(offset, fn=fn):
                rq = ctx.call(fn, q, m + offset, 10000.0, factor)
                rk = ctx.call(fn, k, n + offset, 10000.0, factor)
                check.shape(rq, q.shape, label=fn)
                return np.einsum("bhid,bhjd->bhij", rq, rk)

            check.allclose(
                scores(shift),
                scores(0),
                rtol=1e-3,
                atol=1e-3,
                label=f"{fn}: q.k scores after shifting every position by {shift}",
            )

    check.forall(prop, trials=4)


# Rejects ignores_positions: T = 1 deep into a cached generation.
@case("cached decoding: one new token at position 37")
def position_offset(ctx):
    for fn in BOTH:
        ctx.check(fn, ctx.randn(2, 3, 1, 16), pos(37), 10000.0, 0.5, **TIGHT)


@case("factor = 1: both functions are the half-split RoPE of the whole head")
def factor_one(ctx):
    x = ctx.randn(*SHAPE)
    positions = pos(0, 1, 2, 9, 30)
    a = ctx.check(PARTIAL, x, positions, 500.0, 1.0, **TIGHT)
    b = ctx.check(PROPORTIONAL, x, positions, 500.0, 1.0, **TIGHT)
    check.allclose(a, b, **TIGHT, label="partial_rope versus proportional_rope")


@case("other factors: 0.25, 0.75, and 0.125 of a 16-wide head")
def other_factors(ctx):
    x = ctx.randn(*SHAPE)
    for factor in (0.25, 0.75, 0.125):
        for fn in BOTH:
            ctx.check(fn, x, pos(5, 0, 9, 2, 5), 10000.0, factor, **TIGHT)


# Rejects rounds_pair_count: 0.25 * 12 // 2 = 1 pair (not 2), 0.75 * 12 // 2 = 4 (not 5),
# and 0.375 * 20 // 2 = 3 (not 4).
@case("proportional_rope when factor * D / 2 is not an integer")
def pair_count_floors(ctx):
    x = ctx.randn(2, 3, 5, 12)
    for factor in (0.25, 0.75):
        ctx.check(PROPORTIONAL, x, np.arange(5), 1000.0, factor, **TIGHT)
    ctx.check(PROPORTIONAL, ctx.randn(2, 3, 5, 20), np.arange(5), 1000.0, 0.375, **TIGHT)


@case("position 0 leaves every vector unchanged")
def position_zero(ctx):
    x = ctx.randn(*SHAPE)
    for fn in BOTH:
        got = ctx.check(fn, x, pos(0, 0, 0, 0, 0), 10000.0, 0.5, **TIGHT)
        check.allclose(got, x, label=fn)


# B = H = T = D/4: every broadcastable axis has the same length.
@case("all leading axes the same length: shape (4, 4, 4, 16)")
def all_axes_equal(ctx):
    for fn in BOTH:
        ctx.check(fn, ctx.randn(4, 4, 4, 16), pos(3, 11, 12, 20), 10000.0, 0.5, **TIGHT)


@case("the score of two tokens equals the score with the key rotated by the difference")
def relative_rotation(ctx):
    def prop(rng):
        q, k = ctx.randn(2, 3, 1, 16), ctx.randn(2, 3, 1, 16)
        m, n = sorted(int(v) for v in rng.integers(0, 48, size=2))
        for fn in BOTH:
            got = np.sum(
                ctx.call(fn, q, pos(m), 1000.0, 0.5) * ctx.call(fn, k, pos(n), 1000.0, 0.5), axis=-1
            )
            want = np.sum(q * ctx.call(fn, k, pos(n - m), 1000.0, 0.5), axis=-1)
            check.allclose(
                got, want, rtol=1e-3, atol=1e-3, label=f"{fn}: q.k score of two rotated vectors"
            )

    check.forall(prop, trials=4)


@case("the shipped parameters again, at positions up to 2047")
def shipped_parameters_far(ctx):
    f = ctx.fixture("shipped.npz")
    got = ctx.call(PARTIAL, f["qwen_x"], f["qwen_pos_far"], float(f["qwen_theta"]), 0.25)
    check.allclose(got, f["qwen_y_far"], **LOOSE, label="partial_rope, Qwen3.5 parameters")
    got = ctx.call(PROPORTIONAL, f["gemma_x"], f["gemma_pos_far"], float(f["gemma_theta"]), 0.25)
    check.allclose(got, f["gemma_y_far"], **LOOSE, label="proportional_rope, Gemma 4 parameters")


@case("a small base and a large one: base = 10 and base = 1e7")
def other_bases(ctx):
    x = ctx.randn(*SHAPE)
    for base in (10.0, 1e7):
        for fn in BOTH:
            ctx.check(fn, x, np.arange(5), base, 0.5, **TIGHT)
