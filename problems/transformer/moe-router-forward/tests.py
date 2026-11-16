"""Tests for moe-router-forward. Every case compares the solution with the oracle.

Top-k selection is discontinuous: when the k-th and (k+1)-th router logits of a token
are closer than float32 rounding, two correct solutions can pick different experts.
So the random inputs are drawn until every token has a clear margin at that boundary,
and exact ties are built from small integers, whose products and sums are exact in
float32 in any order of evaluation.
"""

import numpy as np

from learnai.judge import case, check

FN = "moe"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints


def experts(ctx, e, h, d):
    """SwiGLU weights for `e` experts: (w_gate, w_up, w_down), each stacked over experts."""
    return (
        ctx.randn(e, h, d, scale=d**-0.5),
        ctx.randn(e, h, d, scale=d**-0.5),
        ctx.randn(e, d, h, scale=h**-0.5),
    )


def routed(ctx, b, t, d, e, k, scale=1.0, mirror=False):
    """Random `x` and `w_router` such that no token is near a tie at the top-k boundary.

    With `mirror`, the last tokens of each sequence are the negated first ones, so the
    router scores them in the opposite order.
    """
    while True:
        x = ctx.randn(b, t, d)
        if mirror:
            x[:, t // 2 :] = -x[:, : t - t // 2][:, ::-1]
        w_router = ctx.randn(e, d, scale=scale * d**-0.5)
        if k == e:
            return x, w_router
        logits = np.sort(x.astype(np.float64) @ w_router.astype(np.float64).T, axis=-1)
        if np.min(logits[..., e - k] - logits[..., e - k - 1]) > 1e-3:
            return x, w_router


@case("the worked example: three experts, k = 2, and a tie for second place", public=True)
def worked_example(ctx):
    eye = [[1.0, 0.0], [0.0, 1.0]]
    x = np.array([[[2.0, 0.0], [-1.0, -1.0]]], dtype=np.float32)
    w_router = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    w_gate = np.array([eye, [[2.0, 0.0], [0.0, 2.0]], [[-1.0, 0.0], [0.0, -1.0]]], dtype=np.float32)
    w_up = np.array([eye, eye, eye], dtype=np.float32)
    w_down = np.array([eye, [[0.0, 1.0], [1.0, 0.0]], eye], dtype=np.float32)
    got = ctx.check(FN, x, w_router, (w_gate, w_up, w_down), 2, **TOL)
    expected = np.array([[[3.103214, 0.936471], [-0.683633, -0.683633]]], dtype=np.float32)
    check.allclose(got, expected, **TOL, label=FN)


@case("a random layer: B = 2, T = 5, D = 8, four experts of width 12, k = 2", public=True)
def random_layer(ctx):
    x, w_router = routed(ctx, 2, 5, 8, 4, 2)
    ctx.check(FN, x, w_router, experts(ctx, 4, 12, 8), 2, **TOL)


# k = 1 is hard routing (the weight is exactly 1); k = E keeps the full softmax.
@case("the two ends: k = 1 (one expert per token) and k = E (all experts)", public=True)
def k_extremes(ctx):
    for k in (1, 4):
        x, w_router = routed(ctx, 2, 5, 8, 4, k)
        ctx.check(FN, x, w_router, experts(ctx, 4, 12, 8), k, **TOL)


# Rejects ties_to_highest_index. A router of zeros scores every expert 0 for every
# token; integer-valued inputs give exact ties between duplicated router rows.
@case("equal router logits: the expert with the lowest index wins", public=True)
def ties_go_to_lowest_index(ctx):
    weights = experts(ctx, 4, 12, 8)
    x = ctx.randn(2, 5, 8)
    ctx.check(FN, x, np.zeros((4, 8), dtype=np.float32), weights, 2, **TOL)
    # Experts 1 and 3 share a router row, and so do experts 0 and 2.
    x_int = ctx.rng.integers(-2, 3, (2, 5, 8)).astype(np.float32)
    rows = ctx.rng.integers(-2, 3, (2, 8)).astype(np.float32)
    w_router = rows[[0, 1, 0, 1]]
    for k in (1, 3):
        ctx.check(FN, x_int, w_router, weights, k, **TOL)


# If every expert has the same weights, the output is that expert's output whatever the
# routing, because the weights of a token sum to 1. Rejects no_renormalization and
# masks_with_zero without reference to the oracle's routing.
@case("identical experts: the routing weights of every token sum to 1")
def identical_experts(ctx):
    one = experts(ctx, 1, 12, 8)
    same = tuple(np.repeat(w, 4, axis=0) for w in one)
    x, w_router = routed(ctx, 2, 5, 8, 4, 2)
    got = ctx.check(FN, x, w_router, same, 2, **TOL)
    single = ctx.call(FN, x, w_router[:1], one, 1)
    check.allclose(got, single, **TOL, label="moe with four identical experts versus one expert")


# T = E and B = E: a softmax or a top-k along the wrong axis keeps every shape.
@case("as many tokens as experts, and as many sequences as experts: B = T = E = 4")
def square_axes(ctx):
    x, w_router = routed(ctx, 4, 4, 8, 4, 2)
    ctx.check(FN, x, w_router, experts(ctx, 4, 12, 8), 2, **TOL)


# A sharp router: one expert takes almost all the probability, the rest of the top k
# get almost none. Rejects uniform_weights by a wide margin.
@case("a confident router: logits spread over tens of units")
def sharp_router(ctx):
    x, w_router = routed(ctx, 2, 5, 8, 4, 2, scale=25.0)
    got = ctx.check(FN, x, w_router, experts(ctx, 4, 12, 8), 2, **TOL)
    check.finite(got, label=FN)


# Rejects per_sequence_routing: the two halves of each sequence point in opposite
# directions, so they must be sent to different experts.
@case("tokens of one sequence that need different experts")
def per_token_routing(ctx):
    x, w_router = routed(ctx, 2, 5, 8, 4, 1, mirror=True)
    ctx.check(FN, x, w_router, experts(ctx, 4, 12, 8), 1, **TOL)


@case("eight experts with two active, the Mixtral setting, and with three active")
def eight_experts(ctx):
    for k in (2, 3):
        x, w_router = routed(ctx, 2, 5, 8, 8, k)
        ctx.check(FN, x, w_router, experts(ctx, 8, 12, 8), k, **TOL)


@case("a single expert, and a single token")
def single_expert_single_token(ctx):
    x, w_router = routed(ctx, 2, 5, 8, 1, 1)
    ctx.check(FN, x, w_router, experts(ctx, 1, 12, 8), 1, **TOL)
    x, w_router = routed(ctx, 1, 1, 8, 4, 2)
    ctx.check(FN, x, w_router, experts(ctx, 4, 12, 8), 2, **TOL)
