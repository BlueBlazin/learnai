"""Tests for split-merge-heads.

Numeric cases compare with the oracle. Property cases state what the two functions
mean (which input element lands where, and that they are inverses) without it.
Reshaping does no arithmetic, so every comparison is exact.
"""

import numpy as np

from learnai.judge import case, check

EXACT = {"rtol": 0.0, "atol": 0.0}


def counting(*shape):
    """0, 1, 2, ... in the given shape: every element is recognizable in a failure."""
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def check_both(ctx, x, n_heads):
    """split_heads on x, then merge_heads on the oracle's split of x."""
    ctx.check("split_heads", x, n_heads, **EXACT)
    ctx.check("merge_heads", ctx.expected("split_heads", x, n_heads), **EXACT)


@case("the worked example: split a (1, 3, 4) tensor into 2 heads", public=True)
def worked_example(ctx):
    x = counting(1, 3, 4)
    got = ctx.check("split_heads", x, 2, **EXACT)
    expected = np.array(
        [[[[0, 1], [4, 5], [8, 9]], [[2, 3], [6, 7], [10, 11]]]], dtype=np.float32
    )
    check.allclose(got, expected, label="split_heads", **EXACT)


@case("the worked example in reverse: merge 2 heads back into (1, 3, 4)", public=True)
def worked_example_merge(ctx):
    y = np.array([[[[0, 1], [4, 5], [8, 9]], [[2, 3], [6, 7], [10, 11]]]], dtype=np.float32)
    got = ctx.check("merge_heads", y, **EXACT)
    check.allclose(got, counting(1, 3, 4), label="merge_heads", **EXACT)


# T, H, and D / H are all different here, so a reshape to the wrong grouping has the
# wrong shape (reshape_without_transpose) or the wrong values (swaps_heads_and_width,
# split_interleaves_heads, merge_interleaves_heads, merge_transposes_wrong_pair).
@case("a random batch: B = 2, T = 5, D = 12, H = 3", public=True)
def random_batch(ctx):
    check_both(ctx, ctx.randn(2, 5, 12), 3)


# With T == H a reshape to (B, T, H, D / H) has the right shape and the wrong values.
@case("the sequence length equals the number of heads: T = H = 4, D = 24", public=True)
def seq_equals_heads(ctx):
    check_both(ctx, counting(2, 4, 24), 4)
    check_both(ctx, ctx.randn(2, 4, 24), 4)


# Both functions must accept any memory layout. The inputs are built in the solution's
# framework as swaps of two axes, so in PyTorch they are views that are not contiguous
# in memory, and code that calls view where a copy is needed raises.
@case("the input is itself the output of a transpose", public=True)
def transposed_input(ctx):
    x = ctx.randn(2, 5, 12)
    swapped = ctx.fw.xp.swapaxes(ctx.fw.tensor(x.transpose(0, 2, 1).copy()), 1, 2)
    got = ctx.call("split_heads", swapped, 3)
    check.allclose(got, ctx.expected("split_heads", x, 3), label="split_heads", **EXACT)
    y = ctx.randn(2, 3, 5, 4)
    swapped = ctx.fw.xp.swapaxes(ctx.fw.tensor(y.transpose(0, 1, 3, 2).copy()), 2, 3)
    got = ctx.call("merge_heads", swapped)
    check.allclose(got, ctx.expected("merge_heads", y), label="merge_heads", **EXACT)


def random_sizes(rng):
    batch, seq, n_heads, head_width = (int(n) for n in rng.integers(1, 6, size=4))
    return batch, seq, n_heads, head_width


# States the definition without the oracle. Rejects reshape_only, whose round trip is
# the identity although each half is wrong.
@case("head h of token t is the slice x[b, t, h * d : (h + 1) * d]", public=True)
def head_is_a_slice(ctx):
    def prop(rng):
        batch, seq, n_heads, d = random_sizes(rng)
        x = ctx.randn(batch, seq, n_heads * d)
        y = ctx.call("split_heads", x, n_heads)
        check.shape(y, (batch, n_heads, seq, d), label="split_heads")
        for h in range(n_heads):
            check.allclose(
                y[:, h],
                x[:, :, h * d : (h + 1) * d],
                label=f"head {h} of split_heads (of {n_heads} heads of width {d})",
                **EXACT,
            )

    check.forall(prop, trials=5)


@case("merge_heads(split_heads(x)) is x, and split_heads(merge_heads(y)) is y")
def round_trip(ctx):
    def prop(rng):
        batch, seq, n_heads, d = random_sizes(rng)
        # Composed in the solution's framework, so merge_heads receives exactly what
        # split_heads returned (a transposed view, in PyTorch).
        x = ctx.randn(batch, seq, n_heads * d)
        split = ctx.solution.split_heads(ctx.fw.tensor(x), n_heads)
        back = ctx.fw.to_numpy(ctx.solution.merge_heads(split), what="`merge_heads`")
        check.allclose(back, x, label="merge_heads(split_heads(x))", **EXACT)
        y = ctx.randn(batch, n_heads, seq, d)
        merged = ctx.solution.merge_heads(ctx.fw.tensor(y))
        back = ctx.fw.to_numpy(ctx.solution.split_heads(merged, n_heads), what="`split_heads`")
        check.allclose(back, y, label="split_heads(merge_heads(y))", **EXACT)

    check.forall(prop, trials=5)


# A freshly built (contiguous) input. Rejects merge_view_after_transpose, which only
# works on the layout that split_heads returns and so passes the round trip above.
@case("merging concatenates the heads: out[b, t] is head 0, then head 1, ...")
def merge_concatenates(ctx):
    y = ctx.randn(2, 3, 5, 4)
    got = ctx.call("merge_heads", y)
    check.shape(got, (2, 5, 12), label="merge_heads")
    want = np.concatenate([y[:, h] for h in range(3)], axis=-1)
    check.allclose(got, want, label="merge_heads", **EXACT)


@case("a single head, and heads of width 1")
def degenerate_heads(ctx):
    check_both(ctx, ctx.randn(2, 5, 12), 1)
    check_both(ctx, ctx.randn(2, 5, 12), 12)


@case("a single token, and a batch of one")
def single_token(ctx):
    check_both(ctx, ctx.randn(3, 1, 8), 2)
    check_both(ctx, ctx.randn(1, 7, 6), 3)


# Every axis has the same length: only the values can tell the layouts apart.
@case("all axes the same length: B = T = H = D / H = 3")
def all_axes_equal(ctx):
    check_both(ctx, counting(3, 3, 9), 3)
    check_both(ctx, ctx.randn(3, 3, 9), 3)


# Rejects swaps_heads_and_width in both directions (H < D / H and H > D / H).
@case("few wide heads and many narrow heads: H = 2 and H = 8 with D = 16")
def wide_and_narrow(ctx):
    x = ctx.randn(2, 3, 16)
    check_both(ctx, x, 2)
    check_both(ctx, x, 8)
    check_both(ctx, x, 4)
