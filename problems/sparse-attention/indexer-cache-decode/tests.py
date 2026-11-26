"""Tests for indexer-cache-decode.

Every case feeds a sequence to the solution in pieces (`stream`) and checks, after
each call: the outputs against the kit's full causal pass (`TinyDSA.layer_ref`), both
halves of the returned cache against the kit's own keys, and the rows read by the call
against the number of tokens its queries select. At the end it checks that every
token's indexer key was computed exactly once.

The model and the store are handed over behind the kit's `Guarded` (`Layers`, `Store`
below): a solution that touches their counters, private attributes, or `with_framework`
fails, and a call must also have read at least the earlier-call tokens its queries select.

Top-k is discontinuous, so sequences are drawn until no row's k-th and (k+1)-th index
scores are closer than 1e-4 unless they are equal (`clear`); exact ties come from the
ReLU and are settled by the lowest index. Cases written for one rule also assert, on
the fixture, that the rule changes a selection (`self_selected`, `future_would_win`).

The sequence length and the ways of cutting it are fixed, because JAX compiles once
per shape: 8 tokens as one call, as 3 + 1 + 1 + 3, or token by token, and k = 3 except
in one case.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import CountingKV, Guarded, TinyDSA, check_guard, guard

FN = "dsa_step"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
N, K = 8, 3
ONE_CALL, CHUNKS, TOKENS = [8], [3, 4, 5, 8], list(range(1, 9))


class Store(Guarded, CountingKV):
    label = "store"


class Layers(Guarded, TinyDSA):
    label = "dsa"


def pieces(dsa, layer, x):
    """Indexer keys, unmasked index scores (N, N), main keys and values: numpy, uncounted."""
    twin = dsa.with_framework(dsa.fw)
    fw = dsa.fw
    keys = fw.numpy(twin.index_key(layer, fw.tensor(x)))
    q_idx, w = (fw.numpy(t) for t in twin.index_query(layer, fw.tensor(x)))
    per_head = np.maximum(np.einsum("thd,sd->ths", q_idx, keys) * dsa.index_dim**-0.5, 0.0)
    scores = np.einsum("th,ths->ts", w * dsa.index_heads**-0.5, per_head)
    _, k_main, v_main = (fw.numpy(t) for t in twin.qkv(layer, fw.tensor(x)))
    return keys, scores, k_main, v_main


def clear(dsa, layer, x, k, margin=1e-4):
    """No row has its k-th and (k+1)-th causal scores within `margin` unless equal."""
    scores = -np.sort(-dsa.index_scores_ref(layer, x).astype(np.float64), axis=-1)
    if k >= scores.shape[1]:
        return True
    both = np.isfinite(scores[:, k - 1]) & np.isfinite(scores[:, k])
    gap = scores[both, k - 1] - scores[both, k]
    return bool(((gap == 0) | (gap > margin)).all())


def self_selected(dsa, layer, x, k):
    """Some token with more than k visible tokens selects itself."""
    chosen = dsa.indexer_ref(layer, x, k)
    return any(t in chosen[t] for t in range(k, len(x)))


def self_left_out(dsa, layer, x, k):
    """Some token with more than k visible tokens does not select itself."""
    chosen = dsa.indexer_ref(layer, x, k)
    return any(t not in chosen[t] for t in range(k, len(x)))


def future_would_win(dsa, layer, x, k, cuts):
    """In some multi-token call, a later token of the call outscores a selected one."""
    scores = pieces(dsa, layer, x)[1]
    causal = dsa.index_scores_ref(layer, x)
    start = 0
    for stop in cuts:
        for t in range(start, stop - 1):
            kth = np.sort(causal[t, : t + 1])[::-1][min(k, t + 1) - 1]
            if (scores[t, t + 1 : stop] > kth + 1e-3).any():
                return True
        start = stop
    return False


def tie_at_cut(dsa, layer, x, k):
    """Some row has equal scores on both sides of its cut."""
    ranked = -np.sort(-dsa.index_scores_ref(layer, x), axis=-1)
    return bool((ranked[k:, k - 1] == ranked[k:, k]).any())


def draw(ctx, dsa, layer, k, *needs):
    """A sequence with clear selections on which every predicate of `needs` holds."""
    for _ in range(400):
        x = ctx.randn(N, dsa.d_model)
        if clear(dsa, layer, x, k) and all(need(dsa, layer, x, k) for need in needs):
            return x
    raise RuntimeError("authoring error: no fixture with the required properties")


def stream(ctx, dsa, layer, x, cuts, k, public=False):
    """Feed x[: cuts[0]], x[cuts[0] : cuts[1]], ... and check every call. Returns the outputs.

    Failure messages name the call only in public cases.
    """
    want = dsa.layer_ref(layer, x, k)
    want_keys, _, want_k, want_v = pieces(dsa, layer, x)
    dsa.counters.reset()
    guard(ctx)
    first = Store.empty(ctx.fw)
    chosen = dsa.indexer_ref(layer, x, k)
    cache = (np.zeros((0, dsa.index_dim), np.float32), first)
    outs, start = [], 0
    for stop in cuts:
        where = f" (tokens {start} to {stop - 1})" if public else ""
        read_before = first.counters.rows_read
        result = ctx.call(FN, dsa, layer, x[start:stop], cache, k)
        check_guard()
        check.that(
            isinstance(result, tuple) and len(result) == 2,
            f"{FN}: the result is not a pair (out, cache)",
        )
        out, cache = result
        check.shape(out, (stop - start, dsa.d_model), label=f"out{where}")
        check.allclose(out, want[start:stop], **TOL, label=f"out{where}", dtype=True)
        check.that(
            isinstance(cache, tuple) and len(cache) == 2 and isinstance(cache[1], CountingKV),
            f"{FN}: the returned cache is not a pair (index_keys, store)",
        )
        index_keys, store = cache
        check.that(
            isinstance(index_keys, np.ndarray) and index_keys.shape == (stop, dsa.index_dim),
            f"the returned index_keys{where}: got shape {getattr(index_keys, 'shape', None)}, "
            f"expected shape {(stop, dsa.index_dim)} (one indexer key per token so far)",
            f"expected shape {(stop, dsa.index_dim)}, got {getattr(index_keys, 'shape', None)}",
        )
        check.allclose(index_keys, want_keys[:stop], **TOL, label=f"the returned index_keys{where}")
        check.that(
            store.counters is first.counters and store.n == stop,
            f"the returned store{where} was not made by appending the new tokens to the given one",
        )
        got_k, got_v = store.reference()
        check.allclose(got_k, want_k[:stop], **TOL, label=f"the keys in the returned store{where}")
        check.allclose(
            got_v, want_v[:stop], **TOL, label=f"the values in the returned store{where}"
        )
        read = first.counters.rows_read - read_before
        limit = sum(min(k, t + 1) for t in range(start, stop))
        check.that(
            read <= limit,
            f"rows_read{where}: {read} rows in one call exceeds the limit of {limit} rows",
        )
        # Rows of earlier calls exist only in the store: the call must have read every
        # distinct one its queries select.
        earlier = chosen[start:stop]
        needed = len(np.unique(earlier[(earlier >= 0) & (earlier < start)]))
        check.that(
            read >= needed,
            f"rows_read{where}: {read} rows in one call, but its queries select {needed} "
            "tokens of earlier calls, which can only be read from the store",
        )
        outs.append(out)
        start = stop
    check.cost(dsa.counters, "index_key_rows", exactly=len(x), unit="rows")
    limit = sum(min(k, t + 1) for t in range(len(x)))
    check.cost(first.counters, "rows_read", at_most=limit, unit="rows")
    return np.concatenate(outs)


EXAMPLE_X = np.array(
    [[1.0, 0.0, -1.0, 0.5], [0.0, 2.0, 1.0, -1.0], [-1.5, 1.0, 0.0, 1.0]], dtype=np.float32
)


def example_dsa(fw):
    return Layers(
        fw, seed=9, n_layers=1, d_model=4, n_heads=1, head_dim=2, index_heads=2, index_dim=4
    )


@case("the worked example: two tokens, then a third, k = 2", public=True)
def worked_example(ctx):
    dsa = example_dsa(ctx.fw)
    if not clear(dsa, 0, EXAMPLE_X, 2):
        raise RuntimeError("authoring error: the example has a near tie")
    got = stream(ctx, dsa, 0, EXAMPLE_X, [2, 3], 2, public=True)
    expected = np.array(
        [
            [-0.167168, 0.104263, 0.128440, 0.019116],
            [-0.447671, 0.036697, 0.081901, 0.244704],
            [0.178081, 0.025419, 0.010662, -0.129273],
        ],
        dtype=np.float32,
    )
    check.allclose(got, expected, **TOL, label="out")


@case("8 tokens in one call, k = 3", public=True)
def one_call(ctx):
    dsa = Layers(ctx.fw, seed=1)
    x = draw(ctx, dsa, 0, K)
    stream(ctx, dsa, 0, x, ONE_CALL, K, public=True)


# Rejects scores_before_append (a token selects itself although older tokens are
# left out), always_selects_self (another token leaves itself out), reads_all_keys and
# pads_read_as_last_row (cost), and index_cache_not_grown.
@case("8 tokens, one per call, k = 3: at most 3 rows read per token", public=True)
def token_by_token(ctx):
    dsa = Layers(ctx.fw, seed=2)
    x = draw(ctx, dsa, 1, K, self_selected, self_left_out)
    stream(ctx, dsa, 1, x, TOKENS, K, public=True)


# Rejects no_chunk_causality (a later token of a call outscores a selected one),
# last_query_selection, and mask_ignores_cache_offset.
@case("8 tokens as calls of 3, 1, 1, and 3, k = 3", public=True)
def chunked(ctx):
    dsa = Layers(ctx.fw, seed=3)
    x = draw(ctx, dsa, 2, K, lambda *a: future_would_win(*a, CHUNKS))
    stream(ctx, dsa, 2, x, CHUNKS, K, public=True)


@case("the same 8 tokens cut three ways give the same outputs", public=True)
def any_chunking(ctx):
    dsa = Layers(ctx.fw, seed=4)
    x = draw(ctx, dsa, 3, K, self_selected, self_left_out, lambda *a: future_would_win(*a, CHUNKS))
    for cuts in (ONE_CALL, CHUNKS, TOKENS):
        stream(ctx, dsa, 3, x, cuts, K, public=True)


@case(
    "the tensors and the cache passed in are not modified: a cache is continued twice", public=True
)
def inputs_unchanged(ctx):
    dsa = Layers(ctx.fw, seed=5)
    x = draw(ctx, dsa, 0, K)
    want = dsa.layer_ref(0, x, K)
    guard(ctx)
    empty = (ctx.fw.tensor(np.zeros((0, dsa.index_dim), np.float32)), Store.empty(ctx.fw))
    head = ctx.fw.tensor(x[:3])
    _, cache = ctx.solution.dsa_step(dsa, 0, head, empty, K)
    check.allclose(ctx.fw.numpy(head), x[:3], rtol=0, atol=0, label="x_new after the call")
    keys_before = ctx.fw.numpy(cache[0]).copy()
    for _ in range(2):  # the same continuation twice, from the same cache
        out, longer = ctx.solution.dsa_step(dsa, 0, ctx.fw.tensor(x[3:4]), cache, K)
        check.allclose(ctx.fw.numpy(out), want[3:4], **TOL, label="out of a call that continues")
        check.that(longer[1].n == 4 and cache[1].n == 3, "the store passed in changed length")
        check.allclose(
            ctx.fw.numpy(cache[0]), keys_before, rtol=0, atol=0, label="index_keys passed in"
        )
    check_guard()


# Rejects ties_to_highest_index: the ReLU makes scores that are exactly 0, and the
# fixture has a row whose cut falls between equal scores.
@case("equal index scores at the cut")
def tied_scores(ctx):
    dsa = Layers(ctx.fw, seed=6)
    x = draw(ctx, dsa, 1, K, tie_at_cut)
    stream(ctx, dsa, 1, x, CHUNKS, K)
    stream(ctx, dsa, 1, x, TOKENS, K)


@case("k = 1")
def single_selection(ctx):
    dsa = Layers(ctx.fw, seed=7)
    x = draw(ctx, dsa, 2, 1, self_selected)
    stream(ctx, dsa, 2, x, ONE_CALL, 1)


@case("a later token of the same call must not be selected")
def causal_inside_call(ctx):
    dsa = Layers(ctx.fw, seed=8)
    x = draw(ctx, dsa, 0, K, lambda *a: future_would_win(*a, CHUNKS))
    stream(ctx, dsa, 0, x, CHUNKS, K)
    x = draw(ctx, dsa, 1, K, lambda *a: future_would_win(*a, ONE_CALL))
    stream(ctx, dsa, 1, x, ONE_CALL, K)


@case("other layers and seeds")
def random_sweep(ctx):
    def prop(rng):
        dsa = Layers(ctx.fw, seed=int(rng.integers(10, 1000)))
        layer = int(rng.integers(0, 4))
        x = draw(ctx, dsa, layer, K)
        stream(ctx, dsa, layer, x, [ONE_CALL, CHUNKS, TOKENS][int(rng.integers(0, 3))], K)

    check.forall(prop, trials=8)
