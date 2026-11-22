"""Milestone 5 kit, models: LayeredLM, TinyMoE and CountingExperts, TinyEncoder and
CountingScorer, RecurrentLM. Each runs through every installed framework adapter."""

import math

import numpy as np
import pytest

from learnai.judge import frameworks
from learnai.kit import (
    CountingExperts,
    CountingScorer,
    LayerCache,
    LayeredLM,
    RecurrentLM,
    RecurrentState,
    TinyEncoder,
    TinyLM,
    TinyMoE,
    moe_dense_ref,
    moe_route_ref,
)

PROMPT = [5, 9, 12, 40, 7, 33, 2, 18]


@pytest.fixture(params=["torch", "jax", "mlx"])
def fw(request):
    """The tensor frameworks: these models hand out tensors."""
    try:
        return frameworks.get(request.param)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))


def argmax(fw, logits) -> int:
    return int(np.asarray(fw.numpy(logits)).argmax())


# -- LayeredLM ----------------------------------------------------------------------------


def test_layered_lm_is_tiny_lm_with_the_stack_opened(fw):
    lm, plain = LayeredLM(fw, seed=3), TinyLM(fw, seed=3, n_layers=4)
    logits, cache = lm(PROMPT)
    expected, _ = plain(PROMPT)
    assert fw.is_tensor(logits)
    np.testing.assert_array_equal(fw.numpy(logits), fw.numpy(expected))
    assert isinstance(cache, LayerCache) and cache.lengths == (8, 8, 8, 8) and cache.length == 8
    assert lm.counters.snapshot() == {
        "forward_calls": 1, "tokens_processed": 8, "layer_calls": 4, "layer_tokens": 32,
    }  # fmt: skip

    lm.counters.reset()
    h = lm.embed(fw.tensor(np.array(PROMPT)))
    assert fw.is_tensor(h) and fw.numpy(h).shape == (8, 32) and fw.numpy(h).dtype == np.float32
    h, cache = lm.layers(h, 0, 2)
    assert cache.lengths == (8, 8, 0, 0)
    h, cache = lm.layers(h, 2, 4, cache)
    pieces = lm.head(lm.norm(h))
    np.testing.assert_array_equal(fw.numpy(pieces), fw.numpy(expected))
    # embed, norm, and head are free; the two layers calls are what is counted.
    assert lm.counters.snapshot() == {
        "forward_calls": 0, "tokens_processed": 0, "layer_calls": 4, "layer_tokens": 32,
    }  # fmt: skip


def test_layered_lm_continues_a_cache_and_handles_batches(fw):
    lm = LayeredLM(fw, seed=1)
    whole, _ = lm(PROMPT)
    _, cache = lm(PROMPT[:5])
    rest, longer = lm(PROMPT[5:], cache)
    np.testing.assert_allclose(fw.numpy(rest), fw.numpy(whole)[5:], atol=1e-4)
    assert cache.lengths == (5,) * 4 and longer.lengths == (8,) * 4  # the old cache is intact
    short = longer.truncate(5)
    again, _ = lm(PROMPT[5:], short)
    np.testing.assert_allclose(fw.numpy(again), fw.numpy(rest), atol=1e-5)

    batch = np.array([PROMPT, PROMPT[::-1]])
    logits, cache = lm(batch)
    assert fw.numpy(logits).shape == (2, 8, 64) and cache.batch_size == 2 and cache.batched
    np.testing.assert_allclose(fw.numpy(logits)[0], fw.numpy(whole), atol=1e-4)
    swapped = cache.select([1, 0])
    step, _ = lm(np.array([[3], [4]]), swapped)
    direct, _ = lm(np.array([PROMPT[::-1] + [3]]))
    np.testing.assert_allclose(fw.numpy(step)[0, 0], fw.numpy(direct)[0, -1], atol=1e-4)


def test_layered_lm_self_speculation_reuses_the_early_layers(fw):
    """What `early-exit-self-speculation` checks: full-model greedy output, and a
    `layer_tokens` count that recomputing the early layers during verification exceeds."""
    lm = LayeredLM(fw, seed=5, residual_scale=(1, 1, 0.25, 0.25))
    exit_layer, k, new = 2, 3, 12

    def full_greedy(model):
        logits, cache = model(PROMPT)
        out = []
        for _ in range(new):
            out.append(argmax(fw, logits[-1]))
            logits, cache = model([out[-1]], cache)
        return out

    def speculative(model, recompute_early):
        tokens = list(PROMPT)
        h = model.embed(tokens)
        h, cache = model.layers(h, 0, exit_layer)
        early_last = h[-1:]  # exit state of the last token, kept for the next round
        h, cache = model.layers(h, exit_layer, model.n_layers, cache)
        next_token = argmax(fw, model.head(model.norm(h))[-1])
        out = [next_token]
        while len(out) < new:
            base = len(tokens) + len(out) - 1  # tokens the caches cover
            drafts, states, feed = [], [], out[-1]
            for i in range(k):  # draft with the early layers only
                e = model.embed([feed], offset=base + i)
                e, cache = model.layers(e, 0, exit_layer, cache)
                states.append(e)
                feed = argmax(fw, model.head(model.norm(e))[-1])
                drafts.append(feed)
            # Verify: the late layers on the exit states of [out[-1], drafts[:-1]].
            if recompute_early:
                cache = cache.truncate(base)
                e = model.embed([out[-1], *drafts[:-1]], offset=base)
                e, cache = model.layers(e, 0, exit_layer, cache)
            else:
                e = fw.tensor(np.concatenate([np.asarray(fw.numpy(s)) for s in states]))
            v, cache = model.layers(e, exit_layer, model.n_layers, cache)
            verified = np.asarray(fw.numpy(model.head(model.norm(v)))).argmax(-1).tolist()
            accepted = 0
            while accepted < k and drafts[accepted] == verified[accepted]:
                accepted += 1
            keep = verified[: accepted + 1][: new - len(out)]
            out.extend(keep)
            # Roll every layer back to the tokens that are final.
            cache = cache.truncate(len(tokens) + len(out) - 1)
        del early_last
        return out[:new]

    want = full_greedy(lm.with_framework(fw))
    good, bad = lm.with_framework(fw), lm.with_framework(fw)
    assert speculative(good, recompute_early=False) == want
    assert speculative(bad, recompute_early=True) == want
    assert good.counters.layer_tokens < bad.counters.layer_tokens
    # With late layers that only refine, the early exit is usually right.
    assert good.counters.layer_tokens < 0.9 * 4 * (len(PROMPT) + new * 2)


@pytest.mark.parametrize("share_from", [1, 2, 3])
def test_layered_lm_kv_sharing_prefill_costs_the_yoco_formula(fw, share_from):
    """`cross-layer-kv-sharing`: layer_tokens == share_from * P + (L - share_from)."""
    lm = LayeredLM(fw, seed=2)
    oracle = lm.with_framework(fw)
    want, want_cache = oracle(PROMPT, share_from=share_from)
    assert want_cache.lengths == (8,) * share_from + (0,) * (4 - share_from)
    assert oracle.counters.layer_tokens == 4 * 8  # the oracle runs everything on everything

    h = lm.embed(PROMPT)
    h, cache = lm.layers(h, 0, share_from)  # lower layers: every prompt token
    h, cache = lm.layers(h[-1:], share_from, 4, cache, kv_from=share_from - 1)  # upper: last
    logits = lm.head(lm.norm(h))
    np.testing.assert_allclose(fw.numpy(logits)[0], fw.numpy(want)[-1], atol=1e-5)
    assert lm.counters.layer_tokens == share_from * 8 + (4 - share_from)
    assert cache.lengths == want_cache.lengths  # the upper layers store nothing

    # A decode step: one token through the lower layers, then the upper ones read them.
    token = argmax(fw, logits[0])
    h = lm.embed([token], offset=8)
    h, cache = lm.layers(h, 0, share_from, cache)
    h, cache = lm.layers(h, share_from, 4, cache, kv_from=share_from - 1)
    step, _ = oracle([token], want_cache, share_from=share_from)
    np.testing.assert_allclose(fw.numpy(lm.head(lm.norm(h))), fw.numpy(step), atol=1e-5)

    # Upper layers that keep caches of their own compute another model.
    own = lm.with_framework(fw)
    h, _ = own.layers(own.embed(PROMPT), 0, 4)
    assert np.abs(fw.numpy(own.head(own.norm(h)))[-1] - fw.numpy(want)[-1]).max() > 1e-2


def test_layered_lm_refuses_misuse(fw):
    lm = LayeredLM(fw, seed=0)
    h = lm.embed(PROMPT)
    with pytest.raises(ValueError, match="0 <= start < stop <= 4"):
        lm.layers(h, 2, 2)
    with pytest.raises(ValueError, match="0 <= start < stop <= 4"):
        lm.layers(h, 0, 5)
    with pytest.raises(TypeError, match="start must be an int"):
        lm.layers(h, 0.0, 2)
    with pytest.raises(ValueError, match=r"must have shape \(T, 32\)"):
        lm.layers(np.zeros((8, 16), np.float32), 0, 1)
    _, cache = lm.layers(h, 0, 2)
    with pytest.raises(ValueError, match="must continue from the same length"):
        lm.layers(h, 1, 3, cache)
    with pytest.raises(ValueError, match="must be a layer below start"):
        lm.layers(h, 1, 3, cache, kv_from=1)
    with pytest.raises(ValueError, match="run layer 2 on these positions first"):
        lm.layers(h, 3, 4, cache, kv_from=2)
    with pytest.raises(ValueError, match="does not match the cache's 1"):
        lm.layers(lm.embed(np.array([PROMPT, PROMPT])), 0, 2, cache)
    with pytest.raises(ValueError, match="exceeds max_context"):
        lm.embed(PROMPT, offset=125)
    with pytest.raises(ValueError, match=r"share_from must be in \[1, 4\)"):
        lm(PROMPT, share_from=4)
    with pytest.raises(ValueError, match="one number per layer"):
        LayeredLM(fw, residual_scale=(1, 1))
    with pytest.raises(TypeError):
        cache.truncate(1.5)
    assert cache.truncate(3).lengths == (3, 3, 0, 0) and cache.truncate(99).lengths == (8, 8, 0, 0)
    twin = lm.with_framework(fw)
    assert type(twin) is LayeredLM and twin.counters.layer_calls == 0 and lm.counters.layer_calls


# -- CountingExperts and TinyMoE ----------------------------------------------------------


def test_counting_experts_count_calls_and_rows(fw):
    experts = CountingExperts(fw, seed=1, n_experts=4, d_model=8, d_hidden=12)
    x = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
    y = experts(2, fw.tensor(x[:3]))
    assert fw.is_tensor(y) and fw.numpy(y).shape == (3, 8) and fw.numpy(y).dtype == np.float32
    np.testing.assert_allclose(fw.numpy(y), experts.reference(2, x[:3]), rtol=1e-6)
    experts(0, fw.tensor(x))
    experts(3, fw.tensor(x[:0]))  # no rows: still a call
    assert experts.counters.snapshot() == {"expert_calls": 3, "expert_rows": 8}
    assert experts.calls == [(2, 3), (0, 5), (3, 0)]
    # A row's output does not depend on the rows it was batched with.
    np.testing.assert_allclose(
        experts.reference(1, x)[2], experts.reference(1, x[2:3])[0], rtol=1e-5
    )
    assert not np.allclose(experts.reference(0, x), experts.reference(1, x))
    assert [w.shape for w in experts.weights] == [(4, 12, 8), (4, 12, 8), (4, 8, 12)]
    twin = experts.with_framework(fw)
    assert twin.counters.expert_calls == 0 and twin.calls == [] and twin.weights is experts.weights
    with pytest.raises(ValueError, match=r"outside \[0, 4\)"):
        experts(4, fw.tensor(x))
    with pytest.raises(TypeError, match="must be an int"):
        experts("0", fw.tensor(x))
    with pytest.raises(ValueError, match=r"rows of shape \(n, 8\)"):
        experts(0, fw.tensor(x[:, :4]))


def test_moe_sparse_dispatch_matches_the_dense_oracle_at_k_rows_per_token(fw):
    """What `moe-sparse-dispatch` checks: expert_rows == k * n_tokens, at most one
    call per expert that has a token, and the dense oracle's numbers."""
    moe = TinyMoE(fw, seed=2, n_experts=8, top_k=2)
    x = moe.mlp_inputs(np.random.default_rng(1).integers(0, 64, (2, 10)), layer=1)
    assert x.shape == (20, 32) and x.dtype == np.float32
    experts, w_router, k = moe.experts(1), fw.numpy(moe.router(1)), moe.top_k
    assert fw.is_tensor(moe.router(1)) and w_router.shape == (8, 32)
    chosen, weights = moe_route_ref(x, w_router, k)
    assert chosen.shape == weights.shape == (20, 2) and chosen.dtype == np.int64
    np.testing.assert_allclose(weights.sum(-1), 1.0, rtol=1e-6)
    assert (weights[:, 0] >= weights[:, 1]).all()  # largest logit first

    def sparse(model_experts):
        out = np.zeros_like(x)
        for expert in range(8):
            rows, slots = np.nonzero(chosen == expert)
            if len(rows):
                y = np.asarray(fw.numpy(model_experts(expert, fw.tensor(x[rows]))))
                out[rows] += weights[rows, slots][:, None] * y
        return out

    got = sparse(experts)
    want = moe_dense_ref(x, w_router, experts, k)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    assert experts.counters.expert_rows == k * 20
    assert experts.counters.expert_calls <= len(np.unique(chosen))
    # The forward pass and the oracles do not touch the experts' counters.
    before = experts.counters.snapshot()
    logits, _ = moe(PROMPT)
    assert fw.numpy(logits).shape == (8, 64) and experts.counters.snapshot() == before
    assert moe.experts(1) is experts and moe.experts(0) is not experts
    # Ties go to the lowest expert index.
    tied, _ = moe_route_ref(np.ones((1, 4), np.float32), np.ones((3, 4), np.float32), 2)
    assert tied.tolist() == [[0, 1]]
    with pytest.raises(ValueError, match=r"k must be in \[1, 8\]"):
        moe_route_ref(x, w_router, 9)
    with pytest.raises(ValueError, match="top_k"):
        TinyMoE(fw, n_experts=2, top_k=3)
    twin = moe.with_framework(fw)
    twin.experts(1)(0, fw.tensor(x[:2]))
    assert experts.counters.snapshot() == before  # the twin has experts of its own


# -- TinyEncoder and CountingScorer -------------------------------------------------------


def test_encoder_states_do_not_depend_on_padding(fw):
    enc = TinyEncoder(fw, seed=1)
    text = [1, 5, 6, 7]
    alone = enc.reference(text)
    assert alone.shape == (4, 32) and alone.dtype == np.float32
    tokens = np.array([text + [0, 0], [0, 0] + text, [1, 9, 9, 9, 9, 9]])
    mask = np.array([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1], [1] * 6])
    states = enc(fw.tensor(tokens), fw.tensor(mask))
    assert fw.is_tensor(states) and fw.numpy(states).shape == (3, 6, 32)
    got = np.asarray(fw.numpy(states))
    np.testing.assert_allclose(got[0, :4], alone, atol=1e-5)  # right padding
    np.testing.assert_allclose(got[1, 2:], alone, atol=1e-5)  # left padding
    # Bidirectional: changing the last token changes the first token's state.
    other = enc.reference([1, 5, 6, 8])
    assert np.abs(other[0] - alone[0]).max() > 1e-3
    assert enc.counters.snapshot() == {
        "forward_calls": 1,
        "tokens_processed": 18,
        "real_tokens": 14,
    }
    np.testing.assert_array_equal(fw.numpy(enc(text)), alone)
    with pytest.raises(ValueError, match="at least one real"):
        enc(tokens, np.zeros_like(mask))
    with pytest.raises(ValueError, match="shape of tokens"):
        enc(tokens, mask[:, :3])
    with pytest.raises(ValueError, match="exceed max_length"):
        enc(np.ones((1, 65), dtype=np.int64))
    with pytest.raises(ValueError, match=r"in \[0, 64\)"):
        enc([64])
    assert enc.with_framework(fw).counters.forward_calls == 0


def test_scorer_charges_for_the_padded_rectangle_and_scores_pairs_alone():
    """What `rerank-batching` checks: length-sorted batches pad less than arrival-order
    ones, at most ceil(n / max_batch) calls, and scores that survive regrouping."""
    scorer = CountingScorer(seed=1, max_length=24, max_batch=4)
    rng = np.random.default_rng(0)
    query = [3, 4, 5]
    lengths = [2, 19, 3, 20, 2, 18, 4, 21, 3, 17]
    docs = [[int(t) for t in rng.integers(6, 64, n)] for n in lengths]
    pairs = [(query, doc) for doc in docs]
    truth = [scorer.score(query, doc) for doc in docs]
    assert scorer.counters.snapshot() == {
        "calls": 0,
        "pairs": 0,
        "padded_tokens": 0,
        "real_tokens": 0,
    }
    assert all(isinstance(s, float) and -1 < s < 1 for s in truth) and len(set(truth)) == 10

    def run(order, model):
        scores = [0.0] * len(pairs)
        for start in range(0, len(order), 4):
            chunk = order[start : start + 4]
            for index, score in zip(chunk, model([pairs[i] for i in chunk]), strict=True):
                scores[index] = score
        return scores

    arrival, by_length = scorer.with_framework(None), scorer.with_framework(None)
    assert run(list(range(10)), arrival) == truth  # exactly: same floats in any batch
    assert run(sorted(range(10), key=lambda i: lengths[i]), by_length) == truth
    assert arrival.counters.calls == by_length.counters.calls == math.ceil(10 / 4)
    assert arrival.counters.real_tokens == by_length.counters.real_tokens == sum(lengths) + 30
    assert arrival.counters.padded_tokens == 4 * 23 + 4 * 24 + 2 * 20
    assert by_length.counters.padded_tokens == 4 * 6 + 4 * 22 + 2 * 24
    assert by_length.batches == [[5, 5, 6, 6], [7, 20, 21, 22], [23, 24]]
    # The query matters, and so does every document token.
    assert scorer.score([5, 4, 3], docs[0]) != truth[0]
    assert scorer.score(query, docs[1][:-1]) != truth[1]
    for bad, message in [
        ([], "empty batch"),
        ([pairs[0]] * 5, "exceeds max_batch 4"),
        ([(query, [7] * 22)], "truncate the document first"),
        ([(query, [])], "at least one query token and one document token"),
        ([(query, [64])], r"in \[0, 64\)"),
        ([query], "a pair is"),
    ]:
        with pytest.raises(ValueError, match=message):
            scorer(bad)


# -- RecurrentLM --------------------------------------------------------------------------


def test_recurrent_lm_state_is_fixed_size_and_resuming_is_exact(fw):
    lm = RecurrentLM(fw, seed=3)
    tokens = np.random.default_rng(0).integers(0, 64, 40)
    whole, state = lm(fw.tensor(tokens))
    assert fw.is_tensor(whole) and fw.numpy(whole).shape == (40, 64)
    assert isinstance(state, RecurrentState) and state.length == 40
    first, saved = lm(tokens[:17])
    rest, resumed = lm(tokens[17:], saved)
    # Bit for bit, which a KV cache does not give.
    np.testing.assert_array_equal(fw.numpy(rest), fw.numpy(whole)[17:])
    np.testing.assert_array_equal(fw.numpy(first), fw.numpy(whole)[:17])
    assert saved.length == 17 and resumed.length == 40  # `saved` was not modified
    assert saved.nbytes == state.nbytes == 2 * 48 * 4  # the same size at every length
    assert lm.counters.snapshot() == {"forward_calls": 3, "tokens_processed": 80}
    with pytest.raises(TypeError, match="cannot be truncated"):
        state.truncate(10)
    # Resuming from the state of a longer prompt is a different (wrong) answer.
    wrong, _ = lm(tokens[17:20], state)
    assert np.abs(fw.numpy(wrong) - fw.numpy(whole)[17:20]).max() > 0.1

    batch = np.stack([tokens[:6], tokens[6:12]])
    logits, both = lm(batch)
    assert fw.numpy(logits).shape == (2, 6, 64) and both.batch_size == 2
    # Not bit for bit: a batched product may round differently from a single one
    # (it does with the BLAS of Linux on arm64).
    np.testing.assert_allclose(fw.numpy(logits)[0], fw.numpy(whole)[:6], rtol=1e-4, atol=1e-5)
    one = both.select([1])
    step, _ = lm(np.array([[5]]), one)
    alone, _ = lm(np.append(tokens[6:12], 5))
    np.testing.assert_array_equal(fw.numpy(step)[0, 0], fw.numpy(alone)[-1])
    assert lm.initial_state().length == 0 and lm.initial_state(3).batch_size == 3
    for bad, message in [
        (([],), "T >= 1"),
        (([64],), r"in \[0, 64\)"),
        (([1.5],), "must be integers"),
        ((batch, state), "does not match the state's 1"),
        (([1] * 250, saved), "exceeds max_context"),
    ]:
        with pytest.raises(ValueError, match=message):
            lm(*bad)


def test_recurrent_lm_checkpoints_bound_the_tokens_processed(fw):
    """What `hybrid-state-checkpoints` checks:
    tokens_processed <= P - floor(lcp / interval) * interval + g - 1."""
    lm = RecurrentLM(fw, seed=4)
    rng = np.random.default_rng(1)
    first = rng.integers(0, 64, 30).tolist()
    second = first[:21] + rng.integers(0, 64, 9).tolist()
    interval, g = 8, 5

    def generate(model, store, prompt):
        depth = max((n for n in store if n <= len(prompt) and store[n][0] == prompt[:n]), default=0)
        state = store[depth][1] if depth else None
        logits = None
        at = depth
        while at < len(prompt):  # prefill in chunks that end on checkpoint boundaries
            stop = min(len(prompt), (at // interval + 1) * interval)
            logits, state = model(prompt[at:stop], state)
            at = stop
            if at % interval == 0:
                store[at] = (prompt[:at], state)
        out = []
        for _ in range(g):
            out.append(argmax(fw, logits[-1]))
            if len(out) < g:
                logits, state = model([out[-1]], state)
        return out

    store = {}
    a = lm.with_framework(fw)
    out_first = generate(a, store, first)
    assert a.counters.tokens_processed == 30 + g - 1 and sorted(store) == [8, 16, 24]
    b = lm.with_framework(fw)
    out_second = generate(b, store, second)
    lcp = 21
    assert b.counters.tokens_processed == 30 - (lcp // interval) * interval + g - 1
    fresh = lm.with_framework(fw)
    assert generate(fresh, {}, second) == out_second and out_first != out_second
