"""Tests for grouped-query-attention.

Numeric cases compare with the oracle. Property cases state what grouping means
without the oracle: grouped attention equals standard attention on explicitly repeated
keys and values, and a key/value head influences exactly the query heads of its group.
"""

import numpy as np

from learnai.judge import case, check

FN = "gqa"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def causal(t, s=None):
    """Causal mask for the last `t` of `s` positions: True where the key is not in the future."""
    s = t if s is None else s
    query_pos = np.arange(s - t, s)[:, None]
    return np.arange(s)[None, :] <= query_pos


def qkv(ctx, b, h, h_kv, t, s, d, scale=1.0):
    return (
        ctx.randn(b, h, t, d, scale=scale),
        ctx.randn(b, h_kv, s, d, scale=scale),
        ctx.randn(b, h_kv, s, d),
    )


@case("the worked example: H = 4 query heads on H_kv = 2 key/value heads", public=True)
def worked_example(ctx):
    q = np.array([[[[2.0, 0.0]], [[0.0, 2.0]], [[2.0, 0.0]], [[0.0, 2.0]]]], dtype=np.float32)
    k = np.array([[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, -1.0]]]], dtype=np.float32)
    v = np.array([[[[1.0, 0.0], [0.0, 1.0]], [[10.0, 0.0], [0.0, 10.0]]]], dtype=np.float32)
    mask = np.array([[True, True]])
    got = ctx.check(FN, q, k, v, mask, **TOL)
    expected = np.array(
        [[[[0.804430, 0.195570]], [[0.195570, 0.804430]], [[8.044297, 1.955703]],
          [[9.441928, 0.558072]]]],
        dtype=np.float32,
    )  # fmt: skip
    check.allclose(got, expected, atol=1e-5, label=FN)


# Rejects divides_by_kv_head_count (H = 6 is not H_kv squared) and adds_boolean_mask.
@case("a random batch under a causal mask: B = 2, H = 6, H_kv = 2, T = S = 5, d = 8", public=True)
def random_batch(ctx):
    ctx.check(FN, *qkv(ctx, 2, 6, 2, 5, 5, 8), causal(5), **TOL)


# The definition of the grouping, stated without the oracle. Rejects
# tile_instead_of_repeat_interleave and averages_query_heads.
@case("equals standard attention on K and V with every head repeated H / H_kv times", public=True)
def equals_repeated_kv(ctx):
    sizes = [(4, 2), (6, 3), (6, 2), (8, 2)]

    def prop(rng):
        h, h_kv = sizes[int(rng.integers(len(sizes)))]
        q, k, v = qkv(ctx, 2, h, h_kv, 4, 4, 8)
        mask = causal(4)
        grouped = ctx.call(FN, q, k, v, mask)
        check.shape(grouped, (2, h, 4, 8), label=FN)
        groups = h // h_kv
        # np.repeat along the head axis gives the head order 0, 0, .., 1, 1, ..
        full = ctx.call(FN, q, np.repeat(k, groups, axis=1), np.repeat(v, groups, axis=1), mask)
        check.allclose(
            grouped,
            full,
            atol=1e-5,
            label=f"gqa with H = {h}, H_kv = {h_kv} versus gqa on explicitly repeated K and V",
        )

    check.forall(prop, trials=5)


# Rejects divides_by_kv_head_count (out of range for one KV head) and
# scales_by_group_width (fine when H_kv = H, wrong everywhere else).
@case("the two ends: one key/value head for all query heads, and H_kv = H", public=True)
def group_size_extremes(ctx):
    ctx.check(FN, *qkv(ctx, 2, 4, 1, 5, 5, 8), causal(5), **TOL)
    ctx.check(FN, *qkv(ctx, 2, 3, 3, 5, 5, 8), causal(5), **TOL)


@case("eight query heads: H_kv = 4 and H_kv = 2")
def eight_heads(ctx):
    ctx.check(FN, *qkv(ctx, 2, 8, 4, 4, 4, 8), causal(4), **TOL)
    ctx.check(FN, *qkv(ctx, 2, 8, 2, 4, 4, 8), causal(4), **TOL)


@case("cached decoding: T new queries against S cached keys, T < S")
def cached_decoding(ctx):
    # One new token sees the whole cache: the mask is a single row of True.
    ctx.check(FN, *qkv(ctx, 2, 6, 2, 1, 7, 8), causal(1, 7), **TOL)
    # Three new tokens: query t sees the first S - 3 + t + 1 keys.
    ctx.check(FN, *qkv(ctx, 2, 6, 3, 3, 7, 8), causal(3, 7), **TOL)


# Rejects mask_inverted and adds_boolean_mask on a mask that is not triangular.
@case("an arbitrary mask: random allowed positions, at least one per query")
def arbitrary_mask(ctx):
    mask = ctx.rng.random((5, 6)) < 0.5
    mask[np.arange(5), ctx.rng.integers(0, 6, size=5)] = True
    mask[0] = [False, False, False, True, False, False]  # a query that sees a single key
    ctx.check(FN, *qkv(ctx, 2, 4, 2, 5, 6, 8), mask, **TOL)
    ctx.check(FN, *qkv(ctx, 2, 4, 2, 5, 6, 8), ~mask | np.eye(5, 6, dtype=bool), **TOL)


# Rejects tile_instead_of_repeat_interleave and divides_by_kv_head_count directly.
@case("changing one key/value head changes exactly the query heads of its group")
def group_isolation(ctx):
    def prop(rng):
        h_kv, groups = 3, 2
        q, k, v = qkv(ctx, 2, h_kv * groups, h_kv, 4, 4, 8)
        mask = causal(4)
        target = int(rng.integers(h_kv))
        k2, v2 = k.copy(), v.copy()
        k2[:, target] = ctx.randn(2, 4, 8)
        v2[:, target] = ctx.randn(2, 4, 8) + 3.0
        before = ctx.call(FN, q, k, v, mask)
        after = ctx.call(FN, q, k2, v2, mask)
        check.shape(before, (2, 6, 4, 8), label=FN)
        in_group = np.arange(h_kv * groups) // groups == target
        check.allclose(
            after[:, ~in_group],
            before[:, ~in_group],
            atol=1e-5,
            label=f"query heads outside group {target} after changing key/value head {target}",
        )
        moved = np.abs(after[:, in_group] - before[:, in_group]).max(axis=(0, 2, 3))
        check.that(
            bool(np.all(moved > 1e-2)),
            f"a query head of group {target} did not change when key/value head {target} changed",
            f"largest change per query head of the group: {moved}",
        )

    check.forall(prop, trials=3)


# B = H = T = S = d: every axis has the same length, which hides transposed axes.
@case("all axes the same length: B = H = T = S = d = 4, with H_kv = 2 and H_kv = 1")
def all_axes_equal(ctx):
    ctx.check(FN, *qkv(ctx, 4, 4, 2, 4, 4, 4), causal(4), **TOL)
    ctx.check(FN, *qkv(ctx, 4, 4, 1, 4, 4, 4), causal(4), **TOL)


@case("large scores: nearly one-hot attention")
def sharp_attention(ctx):
    got = ctx.check(FN, *qkv(ctx, 2, 6, 2, 5, 5, 8, scale=4.0), causal(5), **TOL)
    check.finite(got, label=FN)


@case("random sizes for B, H, H_kv, T, S, and d")
def random_sizes(ctx):
    for _ in range(2):
        h_kv, groups = int(ctx.rng.integers(1, 5)), int(ctx.rng.integers(1, 5))
        b, t, d = (int(n) for n in ctx.rng.integers(1, 7, size=3))
        s = t + int(ctx.rng.integers(0, 4))
        ctx.check(FN, *qkv(ctx, b, h_kv * groups, h_kv, t, s, d), causal(t, s), **TOL)
