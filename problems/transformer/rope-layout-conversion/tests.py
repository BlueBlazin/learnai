"""Tests for rope-layout-conversion.

Numeric cases compare each function with the oracle. Property cases state what the
conversion is for without the oracle: after `permute_qk`, half-split rotation gives
the attention scores that interleaved rotation gave before. Angles are float32
products of position and frequency, so cases with positions of 64 and above use the
looser tolerance of the statement.
"""

import numpy as np

from whetstone.judge import case, check

TIGHT = {"rtol": 1e-4, "atol": 1e-4}  # positions below 64
LOOSE = {"rtol": 1e-3, "atol": 5e-3}  # positions below 2048
BOTH = ("rope_interleaved", "rope_half")


def pos(*values):
    return np.array(values, dtype=np.int64)


def rows(n, d_in=1):
    """A weight whose row r is filled with the number r: the result shows where rows went."""
    return np.repeat(np.arange(n, dtype=np.float32)[:, None], d_in, axis=1)


def heads(y, n_heads):
    """(B, T, n_heads * d) -> (B, n_heads, T, d): head h owns features h*d .. (h+1)*d - 1."""
    b, t, width = y.shape
    return y.reshape(b, t, n_heads, width // n_heads).transpose(0, 2, 1, 3)


def half_split_order(d_head):
    """Index list p with (P z)[j] = z[p[j]]: even features first, then odd ones."""
    return [*range(0, d_head, 2), *range(1, d_head, 2)]


@case("the worked example: D = 4, theta = 100, and eight rows in two heads", public=True)
def worked_example(ctx):
    x = np.array([[[[1.0, 0.0, 1.0, 0.0]]]], dtype=np.float32)
    got = ctx.check("rope_interleaved", x, pos(1), 100.0, **TIGHT)
    want = [[[[0.540302, 0.841471, 0.995004, 0.099833]]]]
    check.allclose(got, np.array(want, dtype=np.float32), atol=1e-5, label="rope_interleaved")

    got = ctx.check("rope_half", x, pos(1), 100.0, **TIGHT)
    want = [[[[-0.301169, 0.0, 1.381773, 0.0]]]]
    check.allclose(got, np.array(want, dtype=np.float32), atol=1e-5, label="rope_half")

    # The same vector with its features in half-split order: [x0, x2, x1, x3].
    moved = np.array([[[[1.0, 1.0, 0.0, 0.0]]]], dtype=np.float32)
    got = ctx.check("rope_half", moved, pos(1), 100.0, **TIGHT)
    want = [[[[0.540302, 0.995004, 0.841471, 0.099833]]]]
    check.allclose(got, np.array(want, dtype=np.float32), atol=1e-5, label="rope_half")

    got = ctx.check("permute_qk", rows(8), 4, 2, rtol=0.0, atol=0.0)
    want = np.array([0, 2, 1, 3, 4, 6, 5, 7], dtype=np.float32)[:, None]
    check.allclose(got, want, rtol=0.0, atol=0.0, label="permute_qk")


# Rejects half_is_interleaved, half_clockwise, and half_frequencies_by_feature.
@case(
    "random tensors of shape (2, 2, 5, 8): both layouts, positions 0 to 4 and 37 to 41", public=True
)
def random_tensors(ctx):
    for fn in BOTH:
        ctx.check(fn, ctx.randn(2, 2, 5, 8), np.arange(5), **TIGHT)
        ctx.check(fn, ctx.randn(2, 2, 5, 8), 37 + np.arange(5), **TIGHT)
        ctx.check(fn, ctx.randn(2, 2, 5, 8), np.arange(5), theta=500.0, **TIGHT)


# Rejects inverse_permutation: for d_head = 4 the permutation is its own inverse,
# for d_head = 8 it is not.
@case("one head of width 8: rows 0, 2, 4, 6 come first, then rows 1, 3, 5, 7", public=True)
def one_head(ctx):
    got = ctx.check("permute_qk", rows(8, 3), 8, 1, rtol=0.0, atol=0.0)
    want = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.float32)
    check.allclose(got[:, 0], want, rtol=0.0, atol=0.0, label="row order after permute_qk")
    ctx.check("permute_qk", ctx.randn(8, 6), 8, 1, rtol=0.0, atol=0.0)


# Rejects across_heads: every head keeps its own rows.
@case("two heads of width 8: each head is reordered on its own", public=True)
def two_heads(ctx):
    got = ctx.check("permute_qk", rows(16, 3), 8, 2, rtol=0.0, atol=0.0)
    want = np.array([0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15], dtype=np.float32)
    check.allclose(got[:, 0], want, rtol=0.0, atol=0.0, label="row order after permute_qk")
    ctx.check("permute_qk", ctx.randn(16, 6), 8, 2, rtol=0.0, atol=0.0)


# The property the conversion exists for. Rejects every mutant of permute_qk and
# every mutant of rope_half, each for its own reason.
@case("attention scores are identical after conversion", public=True)
def scores_survive_conversion(ctx):
    def prop(rng):
        x = ctx.randn(2, 5, 6)
        w_q, w_k = ctx.randn(16, 6, scale=0.5), ctx.randn(16, 6, scale=0.5)
        positions = np.sort(rng.choice(48, size=5, replace=False))

        def scores(rope, wq, wk):
            q = ctx.call(rope, heads(x @ wq.T, 2), positions)
            k = ctx.call(rope, heads(x @ wk.T, 2), positions)
            check.shape(q, (2, 2, 5, 8), label=rope)
            return np.einsum("bhid,bhjd->bhij", q, k)

        before = scores("rope_interleaved", w_q, w_k)
        p_q, p_k = ctx.call("permute_qk", w_q, 8, 2), ctx.call("permute_qk", w_k, 8, 2)
        check.shape(p_q, (16, 6), label="permute_qk")
        after = scores("rope_half", p_q, p_k)
        check.allclose(
            after,
            before,
            rtol=1e-3,
            atol=1e-3,
            label="q.k scores with permuted weights and rope_half",
        )

    check.forall(prop, trials=5)


@case("rope_half of a reordered vector is the reordered rope_interleaved")
def layouts_commute(ctx):
    order = half_split_order(8)
    x = ctx.randn(2, 2, 5, 8)
    positions = pos(3, 11, 12, 20, 40)
    inter = ctx.call("rope_interleaved", x, positions)
    half = ctx.call("rope_half", x[..., order], positions)
    check.allclose(
        half, inter[..., order], label="rope_half(P x) against P rope_interleaved(x)", **TIGHT
    )
    # And the two are different functions of the same input.
    same_input = ctx.call("rope_half", x, positions)
    check.that(
        float(np.abs(same_input - inter).max()) > 1e-2,
        "rope_half returned the same values as rope_interleaved for the same input",
    )


@case("a single pair (D = 2): the two layouts coincide; and a single token")
def single_pair(ctx):
    x = ctx.randn(2, 2, 5, 2)
    positions = pos(0, 1, 2, 3, 50)
    a = ctx.check("rope_interleaved", x, positions, **TIGHT)
    b = ctx.check("rope_half", x, positions, **TIGHT)
    check.allclose(a, b, atol=1e-6, label="rope_half against rope_interleaved for D = 2")
    for fn in BOTH:
        ctx.check(fn, ctx.randn(1, 1, 1, 6), pos(9), **TIGHT)


@case("position 0 changes nothing; positions out of order and repeated")
def position_edge_cases(ctx):
    x = ctx.randn(2, 2, 5, 8)
    for fn in BOTH:
        got = ctx.check(fn, x, pos(0, 0, 0, 0, 0), **TIGHT)
        check.allclose(got, x, label=f"{fn} at position 0")
        ctx.check(fn, x, pos(5, 0, 9, 2, 5), **TIGHT)


@case("Granite's theta of 1e7 with D = 16, and large positions with D = 32")
def large_theta_and_positions(ctx):
    for fn in BOTH:
        ctx.check(fn, ctx.randn(1, 2, 6, 16), np.arange(6), 1e7, **TIGHT)
        ctx.check(fn, ctx.randn(1, 2, 6, 32), 1000 + np.arange(6), **LOOSE)
        ctx.check(fn, ctx.randn(1, 2, 6, 32), pos(64, 300, 777, 1500, 2000, 2047), **LOOSE)


# Rejects permutes_columns: with a square weight the wrong axis has the right length.
@case("permute_qk: a square weight, heads of width 2 and 6, and four key heads")
def permute_shapes(ctx):
    ctx.check("permute_qk", ctx.randn(16, 16), 8, 2, rtol=0.0, atol=0.0)
    w = ctx.randn(8, 6)
    got = ctx.check("permute_qk", w, 2, 4, rtol=0.0, atol=0.0)
    check.allclose(got, w, rtol=0.0, atol=0.0, label="permute_qk for d_head = 2")
    got = ctx.check("permute_qk", rows(12, 5), 6, 2, rtol=0.0, atol=0.0)
    want = np.array([0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11], dtype=np.float32)
    check.allclose(got[:, 0], want, rtol=0.0, atol=0.0, label="row order after permute_qk")
    ctx.check("permute_qk", ctx.randn(16, 6), 4, 4, rtol=0.0, atol=0.0)
    ctx.check("permute_qk", ctx.randn(32, 6), 32, 1, rtol=0.0, atol=0.0)


@case("grouped-query attention: 4 query heads and 2 key heads, each matrix with its own count")
def grouped_query_conversion(ctx):
    def prop(rng):
        x = ctx.randn(2, 5, 6)
        w_q, w_k = ctx.randn(32, 6, scale=0.5), ctx.randn(16, 6, scale=0.5)
        positions = pos(*(int(p) for p in 7 + np.sort(rng.choice(40, size=5, replace=False))))

        def scores(rope, wq, wk):
            q = ctx.call(rope, heads(x @ wq.T, 4), positions)
            k = ctx.call(rope, heads(x @ wk.T, 2), positions)
            return np.einsum("bhid,bhjd->bhij", q, np.repeat(k, 2, axis=1))

        before = scores("rope_interleaved", w_q, w_k)
        after = scores(
            "rope_half", ctx.call("permute_qk", w_q, 8, 4), ctx.call("permute_qk", w_k, 8, 2)
        )
        check.allclose(after, before, rtol=1e-3, atol=1e-3, label="q.k scores after conversion")

    check.forall(prop, trials=3)
