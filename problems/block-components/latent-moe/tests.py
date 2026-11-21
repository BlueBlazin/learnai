"""Tests for latent-moe.

`dense` below is the layer evaluated densely in numpy from the statement's formulas:
every expert on every token through `experts.reference`, which is not counted. Every
case that calls `latent_moe` goes through `run_layer`, which checks, in this order, the
output against `dense`, `expert_rows` (exactly k * N), `expert_calls` (at most the
number of experts that some token chose), and the call log against the group sizes of
the routing.

The experts are handed over as `Experts`, a `CountingExperts` behind the kit's `Guarded`:
a solution may call them and read their sizes. Their counters, call log, weights, seed,
and `reference` are refused, so the counts cannot be written and the arithmetic cannot
be done anywhere the counts do not see.

Top-k selection is discontinuous, so inputs are drawn until every token has a clear
margin between its k-th and (k+1)-th score (`layer`); exact ties are built from small
integers, whose products and sums are exact in float32 in any order.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import CountingExperts, Guarded, check_guard, guard, ops

FN, COUNT = "latent_moe", "expert_params"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
N, D, L, E, HID = 12, 16, 8, 8, 12  # the sizes most cases share
MARGIN = 1e-3


class Experts(Guarded, CountingExperts):
    """The experts the solution gets: `experts(e, rows)` and the sizes, nothing else."""

    label = "experts"
    refused = Guarded.refused | {"seed", "weights", "calls", "reference"}


def route_np(x, w_router, k):
    """Chosen experts (N, k), best first with ties to the lowest index, and weights."""
    scores = 1.0 / (1.0 + np.exp(-(x.astype(np.float64) @ w_router.astype(np.float64).T)))
    chosen = np.argsort(-scores, axis=-1, kind="stable")[:, :k]
    weights = np.take_along_axis(scores, chosen, axis=-1)
    return chosen, weights / (weights.sum(-1, keepdims=True) + 1e-20), scores


def dense(x, w_router, w_down, w_up, norm_gain, experts, shared_w, k):
    """The statement's formulas, every expert on every token. float64 numpy."""
    chosen, weights, _ = route_np(x, w_router, k)
    z = x @ w_down.T
    mixed = np.zeros(z.shape, dtype=np.float64)
    for j in range(k):
        for e in range(experts.n_experts):
            rows = chosen[:, j] == e
            mixed[rows] += weights[rows, j, None] * experts.reference(e, z[rows])
    normed = norm_gain * mixed / np.sqrt((mixed**2).mean(-1, keepdims=True) + 1e-5)
    return (normed @ w_up.T + ops.swiglu(x, *shared_w)).astype(np.float32)


def layer(ctx, k, n=N, e=E, d=D, latent=L, spread=1.0):
    """Random inputs and parameters of one layer whose routing has a clear margin."""
    while True:
        x = ctx.randn(n, d)
        w_router = ctx.randn(e, d, scale=spread * d**-0.5)
        scores = np.sort(route_np(x, w_router, k)[2], axis=-1)
        if k == e or np.min(scores[:, -k] - scores[:, -k - 1]) > MARGIN:
            break
    return {
        "x": x,
        "w_router": w_router,
        "w_down": ctx.randn(latent, d, scale=d**-0.5),
        "w_up": ctx.randn(d, latent, scale=latent**-0.5),
        "norm_gain": ctx.uniform(latent, low=0.5, high=1.5),
        "shared_w": (
            ctx.randn(HID, d, scale=d**-0.5),
            ctx.randn(HID, d, scale=d**-0.5),
            ctx.randn(d, HID, scale=HID**-0.5),
        ),
    }


def run_layer(ctx, p, experts, k):
    """Call the solution on one layer; check the output, then the work it asked for."""
    args = (p["x"], p["w_router"], p["w_down"], p["w_up"], p["norm_gain"])
    chosen, _, _ = route_np(p["x"], p["w_router"], k)
    sizes = np.bincount(chosen.reshape(-1), minlength=experts.n_experts)
    plan = sorted((e, int(n)) for e, n in enumerate(sizes) if n)

    guard(ctx)
    y = ctx.call(FN, *args, experts, p["shared_w"], k)
    check_guard()
    check.allclose(y, dense(*args, experts, p["shared_w"], k), **TOL, label=FN)
    check.cost(experts.counters, "expert_rows", exactly=k * len(p["x"]), unit="rows")
    check.cost(experts.counters, "expert_calls", at_most=len(plan), unit="calls")
    check.that(
        sorted(experts.calls) == plan,
        f"{FN}: the calls made are not one call per chosen expert with exactly its tokens",
        f"calls made, as (expert, rows): {sorted(experts.calls)}\nexpected: {plan}",
    )
    return y


def latent_experts(ctx, seed, e=E, latent=L):
    return Experts(ctx.fw, seed=seed, n_experts=e, d_model=latent, d_hidden=HID)


@case("the worked example: the parameter counts of Kimi K3.1's routed experts", public=True)
def worked_example(ctx):
    got = ctx.call(COUNT, 7168, 3584, 3072, 896)
    check.equal(list(got), [29646389248, 59190018048], label=COUNT)
    check.equal(list(ctx.call(COUNT, 4, 2, 3, 5)), [106, 180], label=COUNT)


@case("one layer: 12 tokens of width 16, eight experts at width 8, k = 2", public=True)
def one_layer(ctx):
    run_layer(ctx, layer(ctx, 2), latent_experts(ctx, 1), 2)


# The latent output has rows of very different sizes here, so a norm in the wrong
# place, or none, changes the result by far more than the tolerance.
@case("a gain that differs per channel, and k = 3", public=True)
def uneven_gain(ctx):
    p = layer(ctx, 3)
    p["norm_gain"] = np.array([0.25, 2.0, 1.0, 0.5, 3.0, 1.0, 0.125, 1.5], dtype=np.float32)
    p["x"] = p["x"] * np.linspace(0.2, 3.0, N, dtype=np.float32)[:, None]
    run_layer(ctx, p, latent_experts(ctx, 2), 3)


# Three tokens choose at most six of eight experts. Rejects calls_idle_experts, which
# calls the others with zero rows. When every expert is chosen by some token, calling
# every expert is correct, so only a case with idle experts can tell the two apart.
@case("fewer tokens than experts: an expert nobody chose is not called", public=True)
def idle_experts(ctx):
    run_layer(ctx, layer(ctx, 2, n=3), latent_experts(ctx, 3), 2)


@case("parameter counts for other sizes, as Python ints")
def other_counts(ctx):
    for sizes in ((16, 8, 12, 8), (1024, 256, 512, 64), (7168, 7168, 3072, 896), (6, 2, 5, 1)):
        got = ctx.check(COUNT, *sizes)
        check.that(
            all(type(v) is int for v in got),
            f"{COUNT}: the counts are not Python ints",
            f"got {got!r}",
        )


@case("the two ends: k = 2 and k = E")
def k_extremes(ctx):
    for k in (2, E):
        run_layer(ctx, layer(ctx, k), latent_experts(ctx, 5), k)


# Rejects ties_to_highest_index. Experts e and e + 4 share a router row.
@case("equal scores: the expert with the lowest index is chosen")
def tied_scores(ctx):
    p = layer(ctx, 2)
    p["x"] = ctx.rng.integers(-1, 2, (N, D)).astype(np.float32)
    rows = ctx.rng.integers(-1, 2, (4, D)).astype(np.float32)
    p["w_router"] = np.tile(rows, (2, 1))
    for k in (2, 3):
        run_layer(ctx, p, latent_experts(ctx, 6), k)


# A small latent input makes the mixture's mean square comparable to the 1e-5 inside
# the norm. Rejects small_norm_epsilon (1e-6).
@case("small latent activations, where the epsilon of the norm is visible")
def small_inputs(ctx):
    p = layer(ctx, 2)
    p["w_down"] = 0.05 * p["w_down"]
    run_layer(ctx, p, latent_experts(ctx, 7), 2)


@case("a sharp router, more experts than tokens, and a single token")
def sharp_and_small(ctx):
    run_layer(ctx, layer(ctx, 2, spread=6.0), latent_experts(ctx, 8), 2)
    run_layer(ctx, layer(ctx, 3, n=3), latent_experts(ctx, 9), 3)
    run_layer(ctx, layer(ctx, 3, n=1), latent_experts(ctx, 10), 3)
