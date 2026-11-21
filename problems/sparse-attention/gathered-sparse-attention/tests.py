"""Tests for gathered-sparse-attention.

Every case goes through `run`, which checks the output against `dense_ref` (masked
full attention in float64 numpy, the definition) and then the rows read from the
`CountingKV` against the number of valid slots.

The count is the point: a solution that reads every key and masks afterward returns
the same numbers. It is caught wherever the store holds more rows than the call has
valid slots, which is every case except the one that selects every visible token.

The store is handed over as a `Store`, a `CountingKV` behind the kit's `Guarded`: a
solution that touches its counters, its private arrays, or `with_framework` fails.

Shapes are few (JAX compiles once per shape): queries (4, 2, 8) with idx (4, 5), or a
single query (1, 2, 8) with idx (1, 8).
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import CountingKV, Guarded, check_guard, guard

FN = "sparse_attend"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
T, K, H, D, S = 4, 5, 2, 8, 40


class Store(Guarded, CountingKV):
    label = "store"


def dense_ref(q, keys, values, idx):
    """Masked full attention: every key scored, the unselected ones masked out."""
    q, keys, values = (np.asarray(a, np.float64) for a in (q, keys, values))
    selected = np.zeros((q.shape[0], keys.shape[0]), dtype=bool)
    for t, row in enumerate(idx):
        selected[t, row[row >= 0]] = True
    scores = np.einsum("thd,shd->ths", q, keys) / np.sqrt(q.shape[-1])
    scores = np.where(selected[:, None, :], scores, -np.inf)
    top = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - np.where(np.isfinite(top), top, 0.0))
    total = e.sum(axis=-1, keepdims=True)
    weights = e / np.where(total > 0, total, 1.0)  # a query with no token: zeros
    return np.einsum("ths,shd->thd", weights, values).astype(np.float32)


def pick(rng, queries, slots, rows, unused=0.0):
    """Distinct random tokens per query; each slot is unused with probability `unused`."""
    idx = np.stack([rng.permutation(rows)[:slots] for _ in range(queries)]).astype(np.int64)
    idx[rng.random(idx.shape) < unused] = -1
    return idx


def run(ctx, q, keys, values, idx, what=""):
    """Call the solution on a fresh store; check the output, then the rows read."""
    store = Store(ctx.fw, keys, values)
    guard(ctx)
    got = ctx.call(FN, q, store, idx)
    check_guard()
    label = f"{FN}, {what}" if what else FN
    want = dense_ref(q, keys, values, idx)
    check.shape(got, want.shape, label=label)
    check.allclose(got, want, **TOL, label=label, dtype=True)
    check.cost(store.counters, "rows_read", at_most=int((idx >= 0).sum()), unit="rows")
    return got, store


@case("the worked example: two queries, four tokens, one unused slot", public=True)
def worked_example(ctx):
    q = np.array([[[2.0, 0.0]], [[0.0, 2.0]]], np.float32)
    keys = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]], [[1.0, 1.0]]], np.float32)
    values = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[3.0, 3.0]], [[-1.0, 2.0]]], np.float32)
    idx = np.array([[0, 2], [-1, 1]])
    got, store = run(ctx, q, keys, values, idx, "the example")
    expected = np.array([[[1.111614, 0.167422]], [[0.0, 1.0]]], np.float32)
    check.allclose(got, expected, **TOL, label=FN)
    check.cost(store.counters, "rows_read", at_most=3, unit="rows")


@case("4 queries, 5 slots each, a store of 40 tokens: at most 20 rows read", public=True)
def random_selection(ctx):
    q, keys, values = ctx.randn(T, H, D), ctx.randn(S, H, D), ctx.randn(S, H, D)
    run(ctx, q, keys, values, pick(ctx.rng, T, K, S), "every slot used")


# Rejects sentinel_as_index (the last key is the best match of every query),
# unused_scored_zero, masks_after_softmax, and reads_sentinel_rows (cost).
@case("unused slots anywhere in a row: 13 of 20 slots hold a token", public=True)
def unused_slots(ctx):
    q, keys, values = ctx.randn(T, H, D), ctx.randn(S, H, D), ctx.randn(S, H, D)
    keys[-1] = 2.0 * q.mean(axis=0)
    idx = pick(ctx.rng, T, K, S - 1)
    idx[0, [1, 3]] = -1
    idx[1, 0] = -1
    idx[2, [0, 2, 4]] = -1
    idx[3, 4] = -1
    run(ctx, q, keys, values, idx, "13 valid slots")


# Rejects reads_all_then_masks: 64 rows for one query.
@case("a decoding step: one query, 8 of 64 tokens selected, at most 8 rows read", public=True)
def decode_step(ctx):
    q, keys, values = ctx.randn(1, H, D), ctx.randn(64, H, D), ctx.randn(64, H, D)
    run(ctx, q, keys, values, pick(ctx.rng, 1, 8, 64), "one query")


@case("every visible token selected: equal to causal attention", public=True)
def everything_selected(ctx):
    # The queries are the last 4 of 5 tokens; query t lists tokens 0 .. t + 1 in a
    # shuffled order and pads with -1.
    rows = 5
    q, keys, values = ctx.randn(T, H, D), ctx.randn(rows, H, D), ctx.randn(rows, H, D)
    idx = np.full((T, K), -1, dtype=np.int64)
    for t in range(T):
        idx[t, : t + 2] = ctx.rng.permutation(t + 2)
    got, _ = run(ctx, q, keys, values, idx, "all visible tokens")
    visible = np.arange(rows)[None, :] <= np.arange(1, rows)[:, None]
    scores = np.einsum("thd,shd->ths", q, keys).astype(np.float64) / np.sqrt(D)
    scores = np.where(visible[:, None, :], scores, -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    causal = np.einsum("ths,shd->thd", weights, values).astype(np.float32)
    check.allclose(got, causal, **TOL, label=f"{FN} against causal attention")


# Rejects clamps_idx_in_place.
@case("the tensors passed in are not modified", public=True)
def inputs_unchanged(ctx):
    q, keys, values = ctx.randn(T, H, D), ctx.randn(S, H, D), ctx.randn(S, H, D)
    idx = pick(ctx.rng, T, K, S, unused=0.3)
    idx[0, 0] = -1
    store = Store(ctx.fw, keys, values)
    guard(ctx)
    q_t, idx_t = ctx.fw.tensor(q), ctx.fw.tensor(idx)
    ctx.solution.sparse_attend(q_t, store, idx_t)
    check_guard()
    check.allclose(ctx.fw.numpy(q_t), q, rtol=0, atol=0, label="q after the call")
    check.equal(ctx.fw.numpy(idx_t), idx, label="idx after the call")


# Rejects nan_on_empty_row.
@case("a query with no selected token")
def empty_row(ctx):
    q, keys, values = ctx.randn(T, H, D), ctx.randn(S, H, D), ctx.randn(S, H, D)
    idx = pick(ctx.rng, T, K, S, unused=0.2)
    idx[2] = -1
    got, _ = run(ctx, q, keys, values, idx)
    check.finite(got, label=FN)


# Rejects scale_by_k (large scores make the scale visible) and first_row_indices.
@case("sharp attention")
def sharp_scores(ctx):
    q, keys, values = ctx.randn(T, H, D, scale=4.0), ctx.randn(S, H, D), ctx.randn(S, H, D)
    run(ctx, q, keys, values, pick(ctx.rng, T, K, S, unused=0.2))


@case("one token per query")
def single_slot_used(ctx):
    q, keys, values = ctx.randn(T, H, D), ctx.randn(S, H, D), ctx.randn(S, H, D)
    idx = np.full((T, K), -1, dtype=np.int64)
    chosen = ctx.rng.permutation(S)[:T]
    idx[np.arange(T), ctx.rng.integers(0, K, T)] = chosen
    got, _ = run(ctx, q, keys, values, idx)
    check.allclose(got, values[chosen], **TOL, label=FN)


# The same token selected by several queries, and the last and first rows selected.
@case("queries that share tokens")
def shared_tokens(ctx):
    q, keys, values = ctx.randn(T, H, D), ctx.randn(S, H, D), ctx.randn(S, H, D)
    idx = np.array(
        [[S - 1, 0, 7, -1, -1], [0, S - 1, 7, 8, 9], [7, -1, S - 1, -1, 0], [-1, -1, -1, -1, 0]]
    )
    run(ctx, q, keys, values, idx)


@case("a single query over a long store")
def single_query_sweep(ctx):
    def prop(rng):
        q, keys, values = ctx.randn(1, H, D), ctx.randn(64, H, D), ctx.randn(64, H, D)
        run(ctx, q, keys, values, pick(rng, 1, 8, 64, unused=float(rng.choice([0.0, 0.4]))))

    check.forall(prop, trials=5)


@case("random selections")
def random_sweep(ctx):
    def prop(rng):
        scale = float(rng.choice([0.5, 1.0, 3.0]))
        q, keys, values = ctx.randn(T, H, D, scale=scale), ctx.randn(S, H, D), ctx.randn(S, H, D)
        run(ctx, q, keys, values, pick(rng, T, K, S, unused=float(rng.choice([0.0, 0.3, 0.7]))))

    check.forall(prop, trials=8)
