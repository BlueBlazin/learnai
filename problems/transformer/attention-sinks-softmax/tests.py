"""Tests for attention-sinks-softmax.

Numeric cases compare with the oracle. Property cases read the attention weights off
the output by choosing the values: with all-ones values every output component is the
total weight on the real keys, and with one-hot values the output is the weight
vector itself. Those are compared with a float64 computation written here.
"""

import numpy as np

from whetstone.judge import case, check

FN = "attention_with_sinks"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
NEG_INF = -np.inf


def causal(t, s=None):
    s = s or t
    return np.tril(np.ones((t, s), dtype=bool), k=s - t)


def weights_numpy(q, k, mask, sinks):
    """Attention weights over the real keys, (B, H, T, S), in float64."""
    q, k = q.astype(np.float64), k.astype(np.float64)
    scores = q @ np.swapaxes(k, -2, -1) / np.sqrt(q.shape[-1])
    scores = np.where(mask, scores, NEG_INF)
    sink = np.broadcast_to(
        np.asarray(sinks, dtype=np.float64).reshape(1, -1, 1, 1), (*scores.shape[:3], 1)
    )
    logits = np.concatenate([scores, sink], axis=-1)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True))[..., :-1]


def one_hot_values(b, h, s, d):
    """v[b, h, s] is the unit vector e_s, so the output's first S components are the weights."""
    return np.broadcast_to(np.eye(s, d, dtype=np.float32), (b, h, s, d)).copy()


@case(
    "the worked example: three heads with sinks 0, 2, and -inf over the same two keys", public=True
)
def worked_example(ctx):
    q = np.tile(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32), (1, 3, 1, 1))
    k = np.tile(np.array([[1.0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32), (1, 3, 1, 1))
    v = np.tile(np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32), (1, 3, 1, 1))
    mask = np.array([[True, True]])
    sinks = np.array([0.0, 2.0, NEG_INF], dtype=np.float32)
    got = ctx.check(FN, q, k, v, mask, sinks, **TOL)
    expected = np.array(
        [
            [
                [[0.576117, 0.211942, 0.0, 0.0]],
                [[0.244728, 0.090031, 0.0, 0.0]],
                [[0.731059, 0.268941, 0.0, 0.0]],
            ]
        ],
        dtype=np.float32,
    )
    check.allclose(got, expected, atol=1e-5, label=FN)


@case("a random causal layer: B = 2, H = 3, T = S = 5, d = 8", public=True)
def random_causal(ctx):
    q, k, v = ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8)
    ctx.check(FN, q, k, v, causal(5), ctx.randn(3), **TOL)
    ctx.check(FN, q, k, v, causal(5), np.array([1.5, -0.5, 0.25], dtype=np.float32), **TOL)


# Rejects sink_scaled_with_scores, sink_column_masked, and sink_added_to_scores: with
# zero queries every score is 0 and each weight is exactly 1 / (S + exp(sink)).
@case("zero queries: each of S keys gets 1 / (S + exp(sink)), the sink logit unscaled", public=True)
def uniform_scores(ctx):
    q = np.zeros((2, 3, 5, 8), dtype=np.float32)
    k, v = ctx.randn(2, 3, 5, 8), np.ones((2, 3, 5, 8), dtype=np.float32)
    mask = np.ones((5, 5), dtype=bool)
    sinks = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    got = ctx.check(FN, q, k, v, mask, sinks, **TOL)
    total = 5.0 / (5.0 + np.exp(sinks.astype(np.float64)))  # per head
    want = np.broadcast_to(total.reshape(1, 3, 1, 1), got.shape).astype(np.float32)
    check.allclose(got, want, label="total weight on the real keys", **TOL)


# Rejects renormalizes_over_real_keys.
@case(
    "the weights on the real keys sum to less than 1, and to 1 as the sink goes to -inf",
    public=True,
)
def weights_sum_below_one(ctx):
    def prop(rng):
        q, k = ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8)
        ones = np.ones((2, 3, 5, 8), dtype=np.float32)
        mask = causal(5)
        sinks = rng.uniform(-3.0, 3.0, size=3).astype(np.float32)

        def totals(s):
            out = ctx.call(FN, q, k, ones, mask, s)
            check.shape(out, (2, 3, 5, 8), label=FN)
            check.finite(out, label=FN)
            return out

        total = totals(sinks)
        check.that(
            bool(np.all(total < 1.0 - 1e-4)),
            "the weights on the real keys sum to 1 (or more) although the sink is finite",
            f"sinks = {sinks}\nlargest total weight: {float(total.max())}",
        )
        want = weights_numpy(q, k, mask, sinks).sum(axis=-1, keepdims=True)
        check.allclose(total, np.broadcast_to(want, total.shape), label="total weight", **TOL)
        lower = totals(sinks - 4.0)
        check.that(
            bool(np.all(lower > total)),
            "lowering every sink logit did not raise the total weight on the real keys",
        )
        for far in (np.full(3, -80.0, dtype=np.float32), np.full(3, NEG_INF, dtype=np.float32)):
            check.allclose(
                totals(far), ones, atol=1e-5, label=f"total weight with sinks of {far[0]}"
            )

    check.forall(prop, trials=5)


# Rejects sink_shared_by_all_heads, and sink_indexed_by_position because H = T.
@case("one sink per head: H = T = 4 with four different sinks", public=True)
def per_head(ctx):
    q, k, v = ctx.randn(2, 4, 4, 8), ctx.randn(2, 4, 4, 8), ctx.randn(2, 4, 4, 8)
    sinks = np.array([-3.0, 0.0, 1.0, 4.0], dtype=np.float32)
    ctx.check(FN, q, k, v, causal(4), sinks, **TOL)
    ones = np.ones((2, 4, 4, 8), dtype=np.float32)
    total = ctx.call(FN, q, k, ones, causal(4), sinks)
    want = weights_numpy(q, k, causal(4), sinks).sum(axis=-1, keepdims=True)
    check.allclose(
        total, np.broadcast_to(want, total.shape), label="total weight of each head", **TOL
    )


# Rejects drops_wrong_column: the output is the weight vector itself.
@case("one-hot values: the output is the attention weights over the real keys")
def weights_readout(ctx):
    q, k = ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8)
    v = one_hot_values(2, 3, 5, 8)
    sinks = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    for mask in (causal(5), np.ones((5, 5), dtype=bool)):
        got = ctx.check(FN, q, k, v, mask, sinks, **TOL)
        check.allclose(
            got[..., :5], weights_numpy(q, k, mask, sinks), label="attention weights", **TOL
        )
        check.allclose(got[..., 5:], np.zeros_like(got[..., 5:]), label="unused components")


@case("a row with no visible key puts all its weight on the sink: the output is zero")
def fully_masked_row(ctx):
    q, k, v = ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8)
    mask = causal(5)
    mask[0, :] = False
    mask[3, :] = False
    got = ctx.check(FN, q, k, v, mask, ctx.randn(3), **TOL)
    check.finite(got, label=FN)
    check.allclose(
        got[:, :, [0, 3]],
        np.zeros((2, 3, 2, 8), dtype=np.float32),
        label="output of the rows without a visible key",
    )


@case("sinks of -inf give ordinary attention, head by head")
def no_sink(ctx):
    q, k, v = ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8)
    mask = causal(5)
    sinks = np.array([NEG_INF, 0.0, NEG_INF], dtype=np.float32)
    got = ctx.check(FN, q, k, v, mask, sinks, **TOL)
    plain = weights_numpy(q, k, mask, sinks) @ v.astype(np.float64)
    check.allclose(got, plain.astype(np.float32), label=FN, **TOL)
    ctx.check(FN, q, k, v, mask, np.full(3, NEG_INF, dtype=np.float32), **TOL)


@case("cached decoding: one query against five keys, and a sliding-window mask")
def rectangular(ctx):
    k, v = ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8)
    ctx.check(FN, ctx.randn(2, 3, 1, 8), k, v, np.ones((1, 5), dtype=bool), ctx.randn(3), **TOL)
    window = causal(5) & ~np.tril(np.ones((5, 5), dtype=bool), k=-2)
    ctx.check(FN, ctx.randn(2, 3, 5, 8), k, v, window, ctx.randn(3), **TOL)


@case("large scores and large sinks: sharp attention, and a sink that takes almost everything")
def extremes(ctx):
    q, k, v = (
        ctx.randn(2, 3, 5, 8, scale=6.0),
        ctx.randn(2, 3, 5, 8, scale=6.0),
        ctx.randn(2, 3, 5, 8),
    )
    got = ctx.check(FN, q, k, v, causal(5), np.array([50.0, -50.0, 0.0], dtype=np.float32), **TOL)
    check.finite(got, label=FN)
    q, k = ctx.randn(2, 3, 5, 8), ctx.randn(2, 3, 5, 8)
    got = ctx.check(
        FN,
        q,
        k,
        v,
        causal(5),
        np.array([40.0, 60.0, 80.0], dtype=np.float32),
        rtol=1e-4,
        atol=1e-12,
    )
    check.that(float(np.abs(got).max()) < 1e-10, "a sink of 40 or more left weight on the keys")


@case("a single head, a single key, and d = 16")
def small_shapes(ctx):
    ctx.check(
        FN,
        ctx.randn(2, 1, 5, 8),
        ctx.randn(2, 1, 5, 8),
        ctx.randn(2, 1, 5, 8),
        causal(5),
        np.array([0.7], dtype=np.float32),
        **TOL,
    )
    ctx.check(
        FN,
        ctx.randn(2, 3, 1, 8),
        ctx.randn(2, 3, 1, 8),
        ctx.randn(2, 3, 1, 8),
        np.ones((1, 1), dtype=bool),
        ctx.randn(3),
        **TOL,
    )
    ctx.check(
        FN,
        ctx.randn(1, 2, 4, 16),
        ctx.randn(1, 2, 4, 16),
        ctx.randn(1, 2, 4, 16),
        causal(4),
        np.array([2.0, -2.0], dtype=np.float32),
        **TOL,
    )
