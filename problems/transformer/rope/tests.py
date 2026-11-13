"""Tests for rope.

Numeric cases compare with the oracle. Angles are positions times frequencies in
float32, so the error grows with the position: cases with positions below 64 use a tight
tolerance, cases with larger positions a looser one (see the statement's Constraints).
"""

import numpy as np

from whetstone.judge import case, check

FN = "apply_rope"
TIGHT = {"rtol": 1e-4, "atol": 1e-4}  # positions below 64
LOOSE = {"rtol": 1e-3, "atol": 5e-3}  # positions below 2048


def pos(*values):
    return np.array(values, dtype=np.int64)


@case("the worked example: D = 4, base = 100, positions [0, 1]", public=True)
def worked_example(ctx):
    x = np.array([[[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]]], dtype=np.float32)
    got = ctx.check(FN, x, pos(0, 1), base=100.0, **TIGHT)
    expected = np.array(
        [[[[1.0, 0.0, 1.0, 0.0], [0.540302, 0.841471, 0.995004, 0.099833]]]], dtype=np.float32
    )
    check.allclose(got, expected, atol=1e-5, label=FN)


@case("a random tensor of shape (2, 3, 5, 8), positions 0 to 4", public=True)
def random_tensor(ctx):
    ctx.check(FN, ctx.randn(2, 3, 5, 8), np.arange(5), **TIGHT)


# Rejects ignores_positions.
@case("cached decoding: one new token at position 37, then four at 37 to 40", public=True)
def position_offset(ctx):
    ctx.check(FN, ctx.randn(2, 3, 1, 8), pos(37), **TIGHT)
    ctx.check(FN, ctx.randn(2, 3, 4, 8), 37 + np.arange(4), **TIGHT)


@case("attention scores depend only on relative position, and norms are preserved", public=True)
def relative_position(ctx):
    def prop(rng):
        q, k = ctx.randn(2, 2, 4, 8), ctx.randn(2, 2, 4, 8)
        m, n = rng.integers(0, 40, size=4), rng.integers(0, 40, size=4)
        shift = int(rng.integers(1, 24))

        def scores(offset):
            rq = ctx.call(FN, q, m + offset)
            rk = ctx.call(FN, k, n + offset)
            check.shape(rq, q.shape, label=FN)
            return rq, np.einsum("bhid,bhjd->bhij", rq, rk)

        rq, base_scores = scores(0)
        _, shifted_scores = scores(shift)
        check.allclose(
            np.linalg.norm(rq, axis=-1),
            np.linalg.norm(q, axis=-1),
            label="norm of each rotated vector",
        )
        check.allclose(
            shifted_scores,
            base_scores,
            rtol=1e-3,
            atol=1e-3,
            label=f"q.k scores after shifting every position by {shift}",
        )

    check.forall(prop, trials=5)


@case("a different base: 500, and D = 16")
def other_base(ctx):
    ctx.check(FN, ctx.randn(1, 2, 6, 16), np.arange(6), base=500.0, **TIGHT)
    ctx.check(FN, ctx.randn(1, 2, 6, 16), np.arange(6), 500.0, **TIGHT)


@case("a single pair (D = 2) and a single token")
def single_pair(ctx):
    ctx.check(FN, ctx.randn(2, 2, 5, 2), pos(0, 1, 2, 3, 50), **TIGHT)
    ctx.check(FN, ctx.randn(1, 1, 1, 6), pos(9), **TIGHT)


@case("positions out of order and repeated: [5, 0, 9, 2, 5]")
def unordered_positions(ctx):
    ctx.check(FN, ctx.randn(2, 3, 5, 8), pos(5, 0, 9, 2, 5), **TIGHT)


@case("position 0 leaves the vector unchanged")
def position_zero(ctx):
    x = ctx.randn(2, 3, 4, 8)
    got = ctx.check(FN, x, pos(0, 0, 0, 0), **TIGHT)
    check.allclose(got, x, label="apply_rope at position 0")


# B = H = T = D/2: every broadcastable axis has the same length.
@case("all axes the same length: shape (4, 4, 4, 8)")
def all_axes_equal(ctx):
    ctx.check(FN, ctx.randn(4, 4, 4, 8), pos(3, 11, 12, 20), **TIGHT)


@case("the q.k score of two tokens equals the score with k rotated by the position difference")
def relative_rotation(ctx):
    def prop(rng):
        q, k = ctx.randn(2, 3, 1, 8), ctx.randn(2, 3, 1, 8)
        m, n = (int(v) for v in rng.integers(0, 48, size=2))
        if n < m:
            m, n = n, m
        got = np.sum(ctx.call(FN, q, pos(m)) * ctx.call(FN, k, pos(n)), axis=-1)
        want = np.sum(q * ctx.call(FN, k, pos(n - m)), axis=-1)
        check.allclose(got, want, rtol=1e-3, atol=1e-3, label=f"score of positions ({m}, {n})")

    check.forall(prop, trials=5)


@case("large positions: a 6-token window starting at 1000, D = 32")
def large_positions(ctx):
    ctx.check(FN, ctx.randn(1, 2, 6, 32), 1000 + np.arange(6), **LOOSE)
    ctx.check(FN, ctx.randn(1, 2, 6, 32), pos(64, 300, 777, 1500, 2000, 2047), **LOOSE)


@case("random shapes, positions, and bases")
def random_sweep(ctx):
    for _ in range(2):
        b, h, t = (int(n) for n in ctx.rng.integers(1, 5, size=3))
        dim = 2 * int(ctx.rng.integers(1, 9))
        positions = ctx.randint(0, 64, (t,))
        base = float(10.0 ** ctx.rng.uniform(1, 4))
        ctx.check(FN, ctx.randn(b, h, t, dim), positions, base=base, **TIGHT)
