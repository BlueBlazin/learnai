"""Tests for sigmoid-router-shared-expert.

Cases compare with the oracle, with `scores_np` (the two score functions in float64
numpy) where a property can be stated without the oracle, and with the routing weights
that two router modules of `transformers` return for the routing parameters of GLM-5.3
and DeepSeek-V4 (`fixtures/routers.npz`, written by `fixtures/make.py`).

Top-k selection is discontinuous, and `route` also fixes the order of the chosen
experts. So random inputs are drawn until the k + 1 largest biased scores of every
token are well separated (`routed`), and exact ties are built from small integers,
whose products and sums are exact in float32 in any order of evaluation.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import ops

ROUTE, MOE = "route", "moe_shared"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
N, D, E, HID = 6, 8, 8, 12  # the sizes most cases share (JAX compiles once per shape)
MARGIN = 1e-3


def scores_np(x, w_router, score_fn):
    logits = x.astype(np.float64) @ w_router.astype(np.float64).T
    if score_fn == "sigmoid":
        return 1.0 / (1.0 + np.exp(-logits))
    return np.sqrt(np.logaddexp(logits, 0.0))


def routed(ctx, k, score_fn="sigmoid", n=N, e=E, spread=1.0, bias_scale=0.2):
    """Random x, w_router, and bias whose k + 1 best biased scores are well separated."""
    limit = 5.0 if score_fn == "sqrtsoftplus" else 12.0
    while True:
        x = ctx.randn(n, D)
        w_router = ctx.randn(e, D, scale=spread * D**-0.5)
        bias = ctx.randn(e, scale=bias_scale)
        if np.abs(x.astype(np.float64) @ w_router.astype(np.float64).T).max() > limit:
            continue
        ranked = -np.sort(-(scores_np(x, w_router, score_fn) + bias), axis=-1)
        if k == e or np.min(ranked[:, :k] - ranked[:, 1 : k + 1]) > MARGIN:
            return x, w_router, bias


def swiglu_weights(ctx, e, hidden=HID):
    """SwiGLU weights for `e` experts: (w_gate, w_up, w_down), stacked over experts."""
    return (
        ctx.randn(e, hidden, D, scale=D**-0.5),
        ctx.randn(e, hidden, D, scale=D**-0.5),
        ctx.randn(e, D, hidden, scale=hidden**-0.5),
    )


def shared_weights(ctx, hidden=2 * HID):
    return tuple(w[0] for w in swiglu_weights(ctx, 1, hidden))


def dense(indices, weights, n_experts):
    """(N, k) indices and weights as an (N, E) matrix, zero where a token did not choose."""
    out = np.zeros((indices.shape[0], n_experts), dtype=np.float64)
    np.put_along_axis(out, indices.astype(np.int64), weights.astype(np.float64), axis=-1)
    return out


def check_route(ctx, *args, **kwargs):
    """Compare `route` with the oracle: indices exactly, weights within tolerance."""
    indices, weights = ctx.call(ROUTE, *args, **kwargs)
    want_indices, want_weights = ctx.expected(ROUTE, *args, **kwargs)
    check.equal(indices, want_indices, label="route: indices")
    check.allclose(weights, want_weights, **TOL, dtype=True, label="route: weights")
    return indices, weights


EXAMPLE_ROUTER = np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
EXAMPLE_BIAS = np.array([0.0, 0.0, 0.3, 0.0], dtype=np.float32)
EXAMPLE_X = np.array([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32)


@case("the worked example: four experts, k = 2, scale 2.5, a bias on expert 2", public=True)
def worked_example(ctx):
    indices, weights = check_route(ctx, EXAMPLE_X, EXAMPLE_ROUTER, EXAMPLE_BIAS, 2, 2.5, "sigmoid")
    check.equal(indices, [[0, 2], [2, 0]], label="route: indices")
    check.allclose(weights, [[1.594726, 0.905274], [1.25, 1.25]], **TOL, label="route: weights")
    indices, weights = check_route(
        ctx, EXAMPLE_X, EXAMPLE_ROUTER, EXAMPLE_BIAS, 2, 2.5, "sqrtsoftplus"
    )
    check.equal(indices, [[0, 1], [2, 0]], label="route with sqrtsoftplus: indices")
    check.allclose(
        weights,
        [[1.399951, 1.100049], [1.25, 1.25]],
        **TOL,
        label="route with sqrtsoftplus: weights",
    )

    eye, swap = [[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]
    w_gate = np.array([eye, [[2.0, 0.0], [0.0, 2.0]], swap, [[-1.0, 0.0], [0.0, -1.0]]], np.float32)
    w_up = np.array([eye] * 4, np.float32)
    w_down = np.array([eye, eye, [[1.0, 1.0], [0.0, 1.0]], eye], np.float32)
    shared = (
        np.array([[1.0, 1.0]], np.float32),
        np.array([[1.0, -1.0]], np.float32),
        np.array([[1.0], [2.0]], np.float32),
    )
    got = ctx.check(
        MOE, EXAMPLE_X, EXAMPLE_ROUTER, EXAMPLE_BIAS, (w_gate, w_up, w_down), shared, 2, 2.5, **TOL
    )
    check.allclose(got, [[9.141706, 7.046376], [2.741470, 1.827646]], **TOL, label=MOE)


@case("route with sigmoid scores: 6 tokens, 8 experts, k = 3, scale 2.5", public=True)
def route_sigmoid(ctx):
    x, w_router, bias = routed(ctx, 3)
    check_route(ctx, x, w_router, bias, 3, 2.5, "sigmoid")
    check_route(ctx, x, w_router, bias, 3, 2.5)  # sigmoid is the default


@case("route with sqrt(softplus) scores: 6 tokens, 8 experts, k = 3, scale 1.5", public=True)
def route_sqrtsoftplus(ctx):
    x, w_router, bias = routed(ctx, 3, "sqrtsoftplus")
    check_route(ctx, x, w_router, bias, 3, 1.5, "sqrtsoftplus")


@case("the layer: routed experts plus the shared expert", public=True)
def layer(ctx):
    x, w_router, bias = routed(ctx, 3)
    ctx.check(MOE, x, w_router, bias, swiglu_weights(ctx, E), shared_weights(ctx), 3, 2.5, **TOL)


# The library's routers on the routing parameters of two shipped models. The library
# does not order the chosen experts, so the comparison is of dense (N, E) weights.
@case("GLM-5.3 (256 experts, 8 chosen) and DeepSeek-V4 (384, 6): transformers", public=True)
def shipped_routers(ctx):
    f = ctx.fixture("routers.npz")
    for prefix, score_fn, name in (("glm", "sigmoid", "GLM-5.3"), ("ds", "sqrtsoftplus", "V4")):
        k, scale = int(f[f"{prefix}_k"]), float(f[f"{prefix}_scale"])
        indices, weights = check_route(
            ctx, f[f"{prefix}_x"], f[f"{prefix}_w"], f[f"{prefix}_bias"], k, scale, score_fn
        )
        want = f[f"{prefix}_dense"]
        check.allclose(
            dense(indices, weights, want.shape[1]),
            want,
            **TOL,
            label=f"routing weights per expert, {name} parameters",
        )


# Rejects ties_to_highest_index. A router of zeros scores every expert the same;
# duplicated router rows and integer inputs give exact ties elsewhere.
@case("equal biased scores: the expert with the lowest index comes first")
def ties_go_to_lowest_index(ctx):
    x = ctx.randn(N, D)
    zero_bias = np.zeros(E, dtype=np.float32)
    for score_fn in ("sigmoid", "sqrtsoftplus"):
        indices, _ = check_route(ctx, x, np.zeros((E, D), np.float32), zero_bias, 3, 2.5, score_fn)
        check.equal(indices, np.tile(np.arange(3), (N, 1)), label="route: indices")
    # Experts e and e + 4 share a router row and a bias, so they tie exactly. Logits are
    # multiples of 0.5 (exact in float32). Redraw until every pair of biased scores is
    # either exactly equal or more than the stated margin apart.
    bias = np.tile(np.array([0.25, -0.5, 0.0, 0.125], dtype=np.float32), 2)
    while True:
        x_half = 0.5 * ctx.rng.integers(-1, 2, (N, D)).astype(np.float32)
        w_router = np.tile(ctx.rng.integers(-1, 2, (4, D)).astype(np.float32), (2, 1))
        gaps = np.abs(np.diff(np.sort(scores_np(x_half, w_router, "sigmoid") + bias, axis=-1)))
        if gaps[gaps > 0].min() > MARGIN and (gaps == 0).sum() >= 4 * N:
            break
    for k in (1, 3, 5):
        check_route(ctx, x_half, w_router, bias, k, 2.5, "sigmoid")


# Rejects scale_before_normalization, not_normalized, softmax_scores (with the next
# check), and epsilon_too_large without the oracle.
@case("the weights of every token sum to the scale, whatever the score function")
def weights_sum_to_scale(ctx):
    for score_fn, scale in (("sigmoid", 2.5), ("sqrtsoftplus", 0.5), ("sigmoid", 1.0)):
        x, w_router, bias = routed(ctx, 3, score_fn)
        _, weights = ctx.call(ROUTE, x, w_router, bias, 3, scale, score_fn)
        check.shape(weights, (N, 3), label="route: weights")
        check.allclose(
            weights.sum(-1),
            np.full(N, scale),
            rtol=1e-4,
            label="sum of a token's weights",
        )


# The bias moves the selection and must not appear in the weights. Rejects
# weights_from_biased_scores, bias_ignored_in_selection, and bias_added_to_logits.
@case("a large bias decides who is chosen; the weights are ratios of unbiased scores")
def bias_selects_only(ctx):
    for score_fn in ("sigmoid", "sqrtsoftplus"):
        x, w_router, _ = routed(ctx, 3, score_fn, bias_scale=0.0)
        # Experts 5, 6, 7 get a bias no score difference can beat; expert 0 is ruled out.
        bias = np.array([-9.0, 0.0, 0.0, 0.0, 0.0, 5.0, 4.0, 3.0], dtype=np.float32)
        indices, weights = check_route(ctx, x, w_router, bias, 3, 2.0, score_fn)
        check.equal(indices, np.tile([5, 6, 7], (N, 1)), label="route: indices")
        chosen = scores_np(x, w_router, score_fn)[:, 5:]
        check.allclose(
            weights,
            2.0 * chosen / chosen.sum(-1, keepdims=True),
            **TOL,
            label="route: weights",
        )


# Scores near 1e-13: far above 1e-20, so the weights must still sum to the scale.
# Rejects epsilon_too_large. Sigmoid only: float32 softplus is not reliable here.
@case("experts that all score near zero: sigmoid of logits between -32 and -30")
def tiny_scores(ctx):
    # Row e of the router is the constant c[e], so with x = -1 the logit of expert e is
    # exactly -8 * c[e]: -30, -30.25, ..., -31.75 in a shuffled order.
    c = 3.75 + 0.03125 * ctx.rng.permutation(E)
    x = np.full((N, D), -1.0, dtype=np.float32)
    w_router = np.repeat(c[:, None], D, axis=1).astype(np.float32)
    _, weights = check_route(ctx, x, w_router, np.zeros(E, dtype=np.float32), 3, 2.5)
    check.allclose(weights.sum(-1), np.full(N, 2.5), rtol=1e-4, label="sum of a token's weights")


# Logits of -120: every score is exactly 0 in float32, and 0 / (0 + 1e-20) is 0.
# Rejects no_epsilon, which returns NaN.
@case("scores that underflow to zero give weights of zero, not NaN")
def zero_scores(ctx):
    x = np.full((N, D), -15.0, dtype=np.float32)
    w_router = np.ones((E, D), dtype=np.float32)
    bias = np.array([0.0, 0.5, 0.0, 0.25, 0.0, 0.5, 0.0, 0.0], dtype=np.float32)
    for score_fn in ("sigmoid", "sqrtsoftplus"):
        indices, weights = check_route(ctx, x, w_router, bias, 3, 2.5, score_fn)
        check.finite(weights, label="route: weights")
        check.equal(indices, np.tile([1, 5, 3], (N, 1)), label="route: indices")
        check.allclose(weights, np.zeros((N, 3)), atol=1e-12, label="route: weights")


@case("the two ends: k = 1 and k = E")
def k_extremes(ctx):
    for k in (1, E):
        for score_fn in ("sigmoid", "sqrtsoftplus"):
            x, w_router, bias = routed(ctx, k, score_fn)
            check_route(ctx, x, w_router, bias, k, 2.5, score_fn)


# With identical routed experts the routed part is scale * f(x), whatever the routing.
# Rejects shared_in_normalization and shared_expert_scaled without the oracle.
@case("identical routed experts: the output is scale * f(x) + shared(x)")
def identical_experts(ctx):
    one = swiglu_weights(ctx, 1)
    same = tuple(np.repeat(w, E, axis=0) for w in one)
    shared = shared_weights(ctx)
    for score_fn in ("sigmoid", "sqrtsoftplus"):
        x, w_router, bias = routed(ctx, 3, score_fn)
        got = ctx.check(MOE, x, w_router, bias, same, shared, 3, 2.5, score_fn, **TOL)
        want = 2.5 * ops.swiglu(x, *(w[0] for w in one)) + ops.swiglu(x, *shared)
        check.allclose(got, want, **TOL, label=MOE)


@case("routed experts that output zero: the layer is the shared expert alone")
def shared_expert_alone(ctx):
    w_gate, w_up, w_down = swiglu_weights(ctx, E)
    shared = shared_weights(ctx)
    x, w_router, bias = routed(ctx, 3)
    got = ctx.check(MOE, x, w_router, bias, (w_gate, w_up, 0.0 * w_down), shared, 3, 2.5, **TOL)
    check.allclose(got, ops.swiglu(x, *shared), **TOL, label=MOE)


# N = E: a sum or a sort along the wrong axis keeps every shape.
@case("as many tokens as experts, sqrt(softplus) scores, and a single token")
def square_and_single(ctx):
    x, w_router, bias = routed(ctx, 3, "sqrtsoftplus", n=E)
    weights = swiglu_weights(ctx, E)
    shared = shared_weights(ctx)
    ctx.check(MOE, x, w_router, bias, weights, shared, 3, 1.5, "sqrtsoftplus", **TOL)
    check_route(ctx, x, w_router, bias, 3, 1.5, "sqrtsoftplus")
    x, w_router, bias = routed(ctx, 2, n=1)
    ctx.check(MOE, x, w_router, bias, weights, shared, 2, 2.5, score_fn="sigmoid", **TOL)


@case("a score function that is not one of the two names raises ValueError")
def unknown_score_fn(ctx):
    x, w_router, bias = routed(ctx, 3)
    weights, shared = swiglu_weights(ctx, E), shared_weights(ctx)
    for name in ("softmax", "Sigmoid", ""):
        check.raises(ValueError, ctx.call, ROUTE, x, w_router, bias, 3, 2.5, name, label="route")
        check.raises(
            ValueError, ctx.call, MOE, x, w_router, bias, weights, shared, 3, 2.5, name, label=MOE
        )
