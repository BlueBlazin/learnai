"""Tests for moe-sparse-dispatch.

Every case runs one layer through `dispatch`, which checks, in this order: the output
against the kit's dense oracle (`moe_dense_ref`), `expert_rows` (exactly k * N),
`expert_calls` (at most the number of experts that some token selected), and the call
log against the group sizes of the oracle routing (`moe_route_ref`).

The experts are handed over as `Experts`, a `CountingExperts` behind the kit's `Guarded`:
a solution may call them and read their sizes. Their counters, call log, weights, seed,
and `reference` are refused, so the counts cannot be written and the arithmetic cannot
be done anywhere the counts do not see.

Top-k selection is discontinuous, so inputs are drawn until every token has a clear
margin between its k-th and (k+1)-th router logit (`clear`); exact ties are built from
small integers, whose products and sums are exact in float32 in any order.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import (
    CountingExperts,
    Guarded,
    TinyMoE,
    check_guard,
    guard,
    moe_dense_ref,
    moe_route_ref,
)

FN = "moe_sparse"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
MARGIN = 1e-3


class Experts(Guarded, CountingExperts):
    """The experts the solution gets: `experts(e, rows)` and the sizes, nothing else."""

    label = "experts"
    refused = Guarded.refused | {"seed", "weights", "calls", "reference"}


def clear(x, w_router, k, ties=False):
    """Whether rounding cannot change the routing: a margin at the top-k boundary."""
    if k == w_router.shape[0]:
        return True
    logits = np.sort(x.astype(np.float64) @ w_router.astype(np.float64).T, axis=-1)
    gap = logits[:, -k] - logits[:, -k - 1]
    if ties:
        gap = gap[gap != 0]
    return bool((gap > MARGIN).all())


def dispatch(ctx, x, w_router, experts, k, ties=False):
    """Run the solution on one layer and check the output and the work it asked for."""
    x, w_router = x.astype(np.float32), w_router.astype(np.float32)
    if not clear(x, w_router, k, ties):  # an authoring error, not a solution failure
        raise RuntimeError("ambiguous test input: a token is near a tie at the top-k boundary")
    chosen, _ = moe_route_ref(x, w_router, k)
    sizes = np.bincount(chosen.reshape(-1), minlength=experts.n_experts)
    plan = sorted((e, int(n)) for e, n in enumerate(sizes) if n)

    guard(ctx)
    y = ctx.call(FN, x, w_router, Experts.of(experts), k)  # shares the counters and the log
    check_guard()
    check.allclose(y, moe_dense_ref(x, w_router, experts, k), **TOL, label=FN)
    check.cost(experts.counters, "expert_rows", exactly=k * len(x), unit="rows")
    check.cost(experts.counters, "expert_calls", at_most=len(plan), unit="calls")
    check.that(
        sorted(experts.calls) == plan,
        f"{FN}: the calls made are not one call per selected expert with exactly its tokens",
        f"calls made, as (expert, rows): {sorted(experts.calls)}\nexpected: {plan}",
    )
    return y


def layer_inputs(ctx, moe, layer, batch, length):
    """Rows that layer `layer` of the model receives, redrawn until the routing is clear."""
    w_router = ctx.fw.numpy(moe.router(layer))
    while True:
        x = moe.mlp_inputs(ctx.randint(0, 64, (batch, length)), layer)
        if clear(x, w_router, moe.top_k):
            return x, w_router


def random_layer(ctx, n, d, e, k, scale=1.0):
    """Random rows and a random router with a clear routing."""
    while True:
        x = ctx.randn(n, d)
        w_router = ctx.randn(e, d, scale=scale * d**-0.5)
        if clear(x, w_router, k):
            return x, w_router


@case("the worked example: five tokens, five experts, k = 2; expert 4 is idle", public=True)
def worked_example(ctx):
    x = np.array([[3.0, 1.0], [-1.0, 2.0], [2.0, -3.0], [1.0, 3.0], [-3.0, 1.0]], np.float32)
    w_router = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]], np.float32)
    experts = CountingExperts(ctx.fw, seed=1, n_experts=5, d_model=2, d_hidden=4)
    dispatch(ctx, x, w_router, experts, 2)
    check.equal(sorted(experts.calls), [(0, 3), (1, 4), (2, 2), (3, 1)], label="(expert, rows)")


@case("a model's layer: 20 tokens, eight experts, k = 2, uneven groups", public=True)
def model_layer(ctx):
    # Real routing is uneven and interleaved: an expert's tokens are not adjacent.
    # Catches a scatter that assumes they are, and assignment in place of accumulation.
    moe = TinyMoE(ctx.fw, seed=2, n_experts=8, top_k=2)
    x, w_router = layer_inputs(ctx, moe, 1, 2, 10)
    dispatch(ctx, x, w_router, moe.experts(1), 2)


@case("fewer tokens than experts: experts that no token selected are not called", public=True)
def idle_experts(ctx):
    # Three tokens select at most six of eight experts. Catches a loop over all experts
    # that calls each one, with zero rows if need be.
    moe = TinyMoE(ctx.fw, seed=5, n_experts=8, top_k=2)
    x, w_router = layer_inputs(ctx, moe, 0, 1, 3)
    dispatch(ctx, x, w_router, moe.experts(0), 2)


@case(
    "the two ends: k = 1 (one expert per token) and k = E (every expert, every token)", public=True
)
def k_extremes(ctx):
    for k in (1, 4):
        experts = CountingExperts(ctx.fw, seed=3, n_experts=4, d_model=8, d_hidden=12)
        x, w_router = random_layer(ctx, 12, 8, 4, k)
        dispatch(ctx, x, w_router, experts, k)


@case("an expert that is first choice for some tokens and second choice for others")
def mixed_slots(ctx):
    # With three experts and k = 2 every expert appears in both slots. Catches grouping
    # per slot, which calls such an expert twice.
    experts = CountingExperts(ctx.fw, seed=4, n_experts=3, d_model=8, d_hidden=12)
    x, w_router = random_layer(ctx, 12, 8, 3, 2)
    chosen, _ = moe_route_ref(x, w_router, 2)
    if not all((chosen[:, slot] == e).any() for slot in (0, 1) for e in range(3)):
        raise RuntimeError("authoring error: an expert does not appear in both slots")
    dispatch(ctx, x, w_router, experts, 2)


@case("identical tokens: every selected expert receives all of them")
def identical_tokens(ctx):
    experts = CountingExperts(ctx.fw, seed=6, n_experts=8, d_model=8, d_hidden=12)
    x, w_router = random_layer(ctx, 1, 8, 8, 3)
    dispatch(ctx, np.repeat(x, 12, axis=0), w_router, experts, 3)


@case("a single token")
def single_token(ctx):
    experts = CountingExperts(ctx.fw, seed=7, n_experts=8, d_model=8, d_hidden=12)
    x, w_router = random_layer(ctx, 1, 8, 8, 2)
    dispatch(ctx, x, w_router, experts, 2)


@case("equal router logits: the expert with the lowest index is selected")
def tied_logits(ctx):
    # Experts 0 and 2 share a router row, and so do experts 1 and 3; inputs are small
    # integers, so the ties are exact.
    experts = CountingExperts(ctx.fw, seed=8, n_experts=4, d_model=8, d_hidden=12)
    x = ctx.rng.integers(-2, 3, (12, 8)).astype(np.float32)
    rows = ctx.rng.integers(-2, 3, (2, 8)).astype(np.float32)
    for k in (1, 3):
        dispatch(ctx, x, rows[[0, 1, 0, 1]], experts.with_framework(ctx.fw), k, ties=True)


@case("a larger batch through the other layer of the model: 64 tokens, k = 3")
def larger_batch(ctx):
    moe = TinyMoE(ctx.fw, seed=11, n_experts=8, top_k=3)
    x, w_router = layer_inputs(ctx, moe, 0, 4, 16)
    dispatch(ctx, x, w_router, moe.experts(0), 3)
