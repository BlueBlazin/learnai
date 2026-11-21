"""Tests for shared-index-layers.

`run` compares the output with the kit's stack oracle (`TinyDSA.forward_ref`), then the
number of indexer calls with the number of full layers, then, through a log of the
calls (`watch`), which indexer produced the indices each layer attended with.
The model is handed over behind the kit's `Guarded` (`Layers` below): a solution that
touches its counters, private attributes, or `with_framework` fails. The error case must
be raised before any kit call: `TinyDSA.attend` itself raises ValueError for a missing
selection, so without that rule the check would not be tested.

One case compares that sharing pattern with the one `GlmMoeDsaModel` of `transformers`
executes for the layer types of GLM-5.3 (`fixtures/shipped.json`, written by
`fixtures/make.py`).

Top-k is discontinuous, so inputs are drawn until, at every full layer, no row's k-th
and (k+1)-th index scores are closer than 1e-4 unless equal, and until the indexers of
different layers select differently (otherwise sharing could not be told from not
sharing). Every sequence has 10 tokens.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import Guarded, TinyDSA, check_guard, guard

FN = "forward_shared"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
STEPS = 10
F, S = "full", "shared"


class Layers(Guarded, TinyDSA):
    label = "dsa"


def plan(dsa, h, types, k, margin=1e-4):
    """The oracle's selection per full layer, or None if a selection hinges on rounding."""
    rows, idx, chosen = np.asarray(h, np.float32), None, {}
    for layer, kind in enumerate(types):
        if kind == F:
            scores = -np.sort(-dsa.index_scores_ref(layer, rows).astype(np.float64), axis=-1)
            if k < scores.shape[1]:
                both = np.isfinite(scores[:, k - 1]) & np.isfinite(scores[:, k])
                gap = scores[both, k - 1] - scores[both, k]
                if not ((gap == 0) | (gap > margin)).all():
                    return None
            idx = chosen[layer] = dsa.indexer_ref(layer, rows, k)
        rows = rows + dsa.attend_ref(layer, rows, idx)
    return chosen


def draw(ctx, dsa, types, k):
    """An input with clear selections that differ between any two full layers."""
    for _ in range(400):
        h = ctx.randn(STEPS, dsa.d_model)
        chosen = plan(dsa, h, types, k)
        if chosen is None:
            continue
        distinct = {idx.tobytes() for idx in chosen.values()}
        if len(distinct) == len(chosen) or k >= STEPS:
            return h
    raise RuntimeError("authoring error: no input with clear, distinct selections")


def watch(dsa):
    """Log the indexer and attend calls of `dsa`: (name, layer, indices as numpy)."""
    log = []
    indexer, attend = dsa.indexer, dsa.attend

    def logged_indexer(layer, x, k=None):
        idx = indexer(layer, x, k)
        log.append(("indexer", int(layer), np.array(dsa.fw.numpy(idx))))
        return idx

    def logged_attend(layer, x, idx):
        log.append(("attend", int(layer), None if idx is None else np.array(dsa.fw.numpy(idx))))
        return attend(layer, x, idx)

    dsa.indexer, dsa.attend = logged_indexer, logged_attend
    return log


def sources(log, n_layers):
    """For each layer, the layer whose indexer returned the indices it attended with."""
    produced, source = {}, [None] * n_layers
    for name, layer, idx in log:
        if name == "indexer":
            produced[layer] = idx
        elif 0 <= layer < n_layers:
            owners = [
                j for j, own in produced.items() if own.shape == idx.shape and (own == idx).all()
            ]
            source[layer] = max(owners) if owners else None
    return source


def expected_sources(types):
    out, last = [], None
    for layer, kind in enumerate(types):
        last = layer if kind == F else last
        out.append(last)
    return out


def run(ctx, dsa, h, types, k, what=""):
    """Call the solution and check output, indexer calls, and the sharing pattern."""
    want = dsa.forward_ref(h, types, k)
    dsa.counters.reset()
    log = watch(dsa)
    guard(ctx)
    got = ctx.call(FN, dsa, h, list(types), k)
    check_guard()
    label = f"{FN}, {what}" if what else FN
    check.shape(got, want.shape, label=label)
    check.allclose(got, want, **TOL, label=label, dtype=True)
    check.cost(dsa.counters, "indexer_calls", exactly=list(types).count(F), unit="calls")
    used = sources(log, dsa.n_layers)
    check.that(
        used == expected_sources(types),
        f"{label}: a layer attended with indices that are not its nearest full layer's",
        f"layer whose indexer produced each layer's indices: {used}\n"
        f"expected: {expected_sources(types)}",
    )
    return got, used


def check_refuses(ctx, dsa, h, types, label):
    """ValueError, raised by the solution's own check before it calls the kit."""
    log = watch(dsa)
    guard(ctx)
    check.raises(ValueError, ctx.call, FN, dsa, h, types, 3, label=label)
    check_guard()
    check.that(
        not log,
        f"{label}: `dsa.{log[0][0] if log else ''}` was called before the error was raised; "
        "the ValueError must come from your own check, not from the kit refusing a call",
    )


EXAMPLE_H = np.array(
    [[1.0, 0.0, -1.0, 0.5], [0.0, 2.0, 1.0, -1.0], [-1.5, 1.0, 0.0, 1.0], [0.5, -1.0, 2.0, 0.0]],
    dtype=np.float32,
)


@case("the worked example: full, shared, full on four tokens, k = 2", public=True)
def worked_example(ctx):
    dsa = Layers(
        ctx.fw, seed=1, n_layers=3, d_model=4, n_heads=1, head_dim=2, index_heads=2, index_dim=4
    )
    types = [F, S, F]
    if plan(dsa, EXAMPLE_H, types, 2) is None:
        raise RuntimeError("authoring error: the example has a near tie")
    got, _ = run(ctx, dsa, EXAMPLE_H, types, 2, "the example")
    expected = np.array(
        [
            [0.929907, -0.478311, -1.247557, 0.786506],
            [-0.392463, 2.861226, 0.909047, -1.360618],
            [-2.234913, 1.039535, -0.015184, 1.150592],
            [-0.029923, 0.015205, 1.958411, -0.301067],
        ],
        dtype=np.float32,
    )
    check.allclose(got, expected, **TOL, label=FN)


@case("four layers: full, shared, full, shared; 10 tokens, k = 3; 2 indexer calls", public=True)
def alternating(ctx):
    dsa = Layers(ctx.fw, seed=2)
    types = [F, S, F, S]
    run(ctx, dsa, draw(ctx, dsa, types, 3), types, 3, "full, shared, full, shared")


# The layer types of GLM-5.3's first 11 layers, and which layer's indices each layer
# uses when the shipped model runs them.
@case("the first 11 layers of GLM-5.3, as GlmMoeDsaModel shares them: 5 indexer calls", public=True)
def shipped_pattern(ctx):
    f = ctx.fixture("shipped.json")
    types = f["types"]
    dsa = Layers(ctx.fw, seed=3, n_layers=len(types))
    _, used = run(ctx, dsa, draw(ctx, dsa, types, 3), types, 3, "GLM-5.3 layer types")
    check.equal(used, f["source"], label="the layer whose indices each layer used")
    check.cost(dsa.counters, "indexer_calls", exactly=f["indexer_calls"], unit="calls")


@case("a shared layer with no full layer before it raises ValueError", public=True)
def shared_first(ctx):
    dsa = Layers(ctx.fw, seed=4)
    h = ctx.randn(STEPS, dsa.d_model)
    for types in ([S, F, S, F], [S, S, S, S], [S, S, F, F]):
        check_refuses(ctx, dsa, h, types, f"{FN} with types {types}")


# Rejects in_place_residual.
@case("h is not modified", public=True)
def inputs_unchanged(ctx):
    dsa = Layers(ctx.fw, seed=5)
    types = [F, S, S, F]
    h = draw(ctx, dsa, types, 3)
    tensor = ctx.fw.tensor(h)
    guard(ctx)
    got = ctx.fw.numpy(ctx.solution.forward_shared(dsa, tensor, types, 3))
    check_guard()
    check.allclose(ctx.fw.numpy(tensor), h, rtol=0, atol=0, label="h after the call")
    check.allclose(got, dsa.forward_ref(h, types, 3), **TOL, label=FN)


@case("every layer full: one indexer call per layer", public=True)
def all_full(ctx):
    dsa = Layers(ctx.fw, seed=6)
    types = [F, F, F, F]
    run(ctx, dsa, draw(ctx, dsa, types, 3), types, 3, "four full layers")


# Rejects reuses_first_full (layers 3 and 4 follow layer 2, not layer 0) and
# uses_dsa_types (the kit's own default pattern is full, shared, full, shared, ...).
@case("shared layers after a second full layer")
def nearest_not_first(ctx):
    dsa = Layers(ctx.fw, seed=7, n_layers=5)
    types = [F, S, F, S, S]
    run(ctx, dsa, draw(ctx, dsa, types, 3), types, 3)


@case("one full layer, then shared layers only")
def single_full(ctx):
    dsa = Layers(ctx.fw, seed=8, n_layers=6)
    types = [F, S, S, S, S, S]
    run(ctx, dsa, draw(ctx, dsa, types, 3), types, 3)


# Rejects ignores_k: the kit's default selection size is 4.
@case("other selection sizes")
def other_k(ctx):
    dsa = Layers(ctx.fw, seed=9)
    types = [F, F, S, S]
    for k in (1, 2, 6, 25):
        run(ctx, dsa, draw(ctx, dsa, types, k), types, k)


@case("a shared first layer, with other layer counts")
def shared_first_hidden(ctx):
    h = ctx.randn(STEPS, 32)
    for types in ([S], [S, F], [S, S, S, F, S, F]):
        dsa = Layers(ctx.fw, seed=10, n_layers=len(types))
        check_refuses(ctx, dsa, h, types, FN)
    dsa = Layers(ctx.fw, seed=10, n_layers=1)
    run(ctx, dsa, draw(ctx, dsa, [F], 3), [F], 3)


@case("random layer types")
def random_types(ctx):
    def prop(rng):
        n_layers = int(rng.choice([4, 6]))
        types = [F] + [F if rng.random() < 0.4 else S for _ in range(n_layers - 1)]
        dsa = Layers(ctx.fw, seed=int(rng.integers(20, 1000)), n_layers=n_layers)
        run(ctx, dsa, draw(ctx, dsa, types, 3), types, 3)

    check.forall(prop, trials=8)
