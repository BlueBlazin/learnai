"""KVCache.truncate / KVCache.select and draft_target_pair."""

import numpy as np
import pytest

from whetstone.judge import frameworks
from whetstone.kit import KVCache, TinyLM, draft_target_pair

PROMPT = [0, 5, 9, 33, 2, 7]
PY = frameworks.get("python")


def logits_of(value):
    return np.asarray(value, dtype=np.float32)


# -- truncate --------------------------------------------------------------------------


def test_truncate_rolls_back_to_a_prefix():
    lm = TinyLM(PY, seed=2)
    _, cache = lm(PROMPT)
    short = cache.truncate(3)
    assert isinstance(short, KVCache) and short.length == 3 and cache.length == len(PROMPT)
    # Continuing from the truncated cache equals continuing from a cache of the prefix.
    _, prefix_cache = lm(PROMPT[:3])
    got, got_cache = lm([40, 41], short)
    want, _ = lm([40, 41], prefix_cache)
    np.testing.assert_allclose(logits_of(got), logits_of(want), rtol=1e-5, atol=1e-5)
    assert got_cache.length == 5


def test_truncate_to_the_full_length_and_to_zero():
    lm = TinyLM(PY, seed=2)
    _, cache = lm(PROMPT)
    same = cache.truncate(cache.length)
    assert same.length == cache.length
    np.testing.assert_array_equal(same.keys[0], cache.keys[0])
    empty = cache.truncate(0)
    assert empty.length == 0 and empty.keys[0].shape[2] == 0
    # An empty cache is the same as no cache.
    from_empty, new_cache = lm(PROMPT, empty)
    fresh, _ = lm(PROMPT)
    np.testing.assert_array_equal(logits_of(from_empty), logits_of(fresh))
    assert new_cache.length == len(PROMPT)


def test_truncate_keeps_the_batch_and_never_copies_or_mutates():
    lm = TinyLM(PY, seed=1)
    _, cache = lm([PROMPT, PROMPT[::-1]])
    short = cache.truncate(2)
    assert short.batch_size == 2 and short.batched
    assert np.shares_memory(short.keys[0], cache.keys[0])
    assert not short.keys[0].flags.writeable and not short.values[1].flags.writeable
    assert cache.length == len(PROMPT) and cache.keys[0].shape[2] == len(PROMPT)


@pytest.mark.parametrize("n", [-1, 7, 100])
def test_truncate_rejects_lengths_outside_the_cache(n):
    _, cache = TinyLM(PY)(PROMPT)
    with pytest.raises(ValueError, match="cannot truncate"):
        cache.truncate(n)


@pytest.mark.parametrize("n", [2.0, "2", True, None])
def test_truncate_rejects_non_integers(n):
    _, cache = TinyLM(PY)(PROMPT)
    with pytest.raises(TypeError):
        cache.truncate(n)


def test_truncate_accepts_numpy_integers():
    _, cache = TinyLM(PY)(PROMPT)
    assert cache.truncate(np.int64(4)).length == 4


# -- select ----------------------------------------------------------------------------


def test_select_reorders_rows():
    lm = TinyLM(PY, seed=3)
    rows = [PROMPT, PROMPT[::-1], [1, 2, 3, 4, 5, 6]]
    _, cache = lm(rows)
    picked = cache.select([2, 0])
    assert picked.batch_size == 2 and picked.length == cache.length
    nxt = [[10], [11]]
    got, _ = lm(nxt, picked)
    _, direct = lm([rows[2], rows[0]])
    want, _ = lm(nxt, direct)
    np.testing.assert_allclose(logits_of(got), logits_of(want), rtol=1e-5, atol=1e-5)


def test_select_repeats_rows_and_expands_an_unbatched_cache():
    lm = TinyLM(PY, seed=3)
    _, cache = lm(PROMPT)  # 1-D tokens: one row, not batched
    assert cache.batch_size == 1 and not cache.batched
    beams = cache.select([0, 0, 0])
    assert beams.batch_size == 3 and beams.batched
    got, _ = lm([[8], [8], [9]], beams)
    got = logits_of(got)
    np.testing.assert_array_equal(got[0], got[1])  # the same row twice gives the same logits
    single, _ = lm([9], cache)
    np.testing.assert_allclose(got[2, 0], logits_of(single)[0], rtol=1e-5, atol=1e-5)
    assert not np.array_equal(got[0], got[2])


def test_select_copies_and_leaves_the_source_alone():
    _, cache = TinyLM(PY)([PROMPT, PROMPT[::-1]])
    before = cache.keys[0].copy()
    picked = cache.select([1, 1, 0])
    assert not np.shares_memory(picked.keys[0], cache.keys[0])
    assert not picked.keys[0].flags.writeable
    np.testing.assert_array_equal(cache.keys[0], before)
    assert cache.batch_size == 2


def test_select_accepts_tensors(fw):
    lm = TinyLM(fw, seed=1)
    _, cache = lm(np.array([PROMPT, PROMPT[::-1]]))
    picked = cache.select(fw.tensor(np.array([1, 0, 1])))
    assert picked.batch_size == 3
    np.testing.assert_array_equal(picked.keys[0][0], cache.keys[0][1])


@pytest.mark.parametrize("rows", [[], [2], [-1], [0.0], [[0, 1]], 0])
def test_select_rejects_bad_rows(rows):
    _, cache = TinyLM(PY)([PROMPT, PROMPT])
    with pytest.raises(ValueError):
        cache.select(rows)


def test_select_then_truncate_compose():
    lm = TinyLM(PY, seed=5)
    _, cache = lm([PROMPT, PROMPT[::-1]])
    a = cache.select([1, 0]).truncate(4)
    b = cache.truncate(4).select([1, 0])
    assert a.length == b.length == 4
    for x, y in zip((*a.keys, *a.values), (*b.keys, *b.values), strict=True):
        np.testing.assert_array_equal(x, y)


# -- draft_target_pair -------------------------------------------------------------------


def greedy_agreement(draft, target, seed, prompts=20, prompt_len=5, new_tokens=30):
    """Fraction of steps, along the target's greedy decode of fresh prompts, at which
    the draft's greedy token equals the target's."""
    rng = np.random.default_rng(1000 + seed)
    hits = total = 0
    for _ in range(prompts):
        tokens = [int(t) for t in rng.integers(0, target.vocab_size, prompt_len)]
        t_logits, t_cache = target(tokens)
        d_logits, d_cache = draft(tokens)
        for _ in range(new_tokens):
            t_next, d_next = int(np.argmax(t_logits[-1])), int(np.argmax(d_logits[-1]))
            hits += t_next == d_next
            total += 1
            t_logits, t_cache = target([t_next], t_cache)
            d_logits, d_cache = draft([t_next], d_cache)
    return hits / total


@pytest.mark.parametrize("agreement", [0.5, 0.8, 0.9])
def test_greedy_agreement_is_near_the_requested_rate_across_seeds(agreement):
    rates = []
    for seed in range(6):
        draft, target = draft_target_pair(PY, seed, agreement)
        rates.append(greedy_agreement(draft, target, seed))
    # Each pair is close on prompts it was not tuned on, and the average is closer.
    assert all(abs(rate - agreement) <= 0.10 for rate in rates), rates
    assert abs(np.mean(rates) - agreement) <= 0.04, rates


def test_agreement_one_gives_identical_models_and_lower_rates_differ():
    draft, target = draft_target_pair(PY, 4, agreement=1.0)
    np.testing.assert_array_equal(logits_of(draft(PROMPT)[0]), logits_of(target(PROMPT)[0]))
    draft, target = draft_target_pair(PY, 4, agreement=0.7)
    assert not np.array_equal(logits_of(draft(PROMPT)[0]), logits_of(target(PROMPT)[0]))


def test_the_pair_is_deterministic_and_the_target_is_the_plain_model():
    a_draft, a_target = draft_target_pair(PY, 7, 0.8)
    b_draft, b_target = draft_target_pair(PY, 7, 0.8)
    np.testing.assert_array_equal(logits_of(a_draft(PROMPT)[0]), logits_of(b_draft(PROMPT)[0]))
    np.testing.assert_array_equal(logits_of(a_target(PROMPT)[0]), logits_of(b_target(PROMPT)[0]))
    plain = TinyLM(PY, seed=7)
    np.testing.assert_array_equal(logits_of(a_target(PROMPT)[0]), logits_of(plain(PROMPT)[0]))
    other, _ = draft_target_pair(PY, 8, 0.8)
    assert not np.array_equal(logits_of(other(PROMPT)[0]), logits_of(a_draft(PROMPT)[0]))


def test_the_models_start_with_zero_counters_and_count_separately():
    draft, target = draft_target_pair(PY, 1, 0.8)
    assert draft.counters.snapshot() == target.counters.snapshot() == {
        "forward_calls": 0,
        "tokens_processed": 0,
    }
    draft(PROMPT)
    assert draft.counters.tokens_processed == len(PROMPT)
    assert target.counters.tokens_processed == 0


def test_the_pair_speaks_the_framework_and_shares_a_vocabulary(fw):
    SMALL = [0, 5, 9, 31, 2, 7]
    draft, target = draft_target_pair(fw, 2, 0.8, vocab_size=32, max_context=48)
    assert draft.vocab_size == target.vocab_size == 32
    assert draft.max_context == target.max_context == 48
    assert draft.fw is fw and target.fw is fw
    logits, cache = draft(SMALL)
    assert cache.length == len(SMALL)
    if fw.name != "python":
        assert fw.is_tensor(logits)
    # Rebinding (what ctx.expected does) keeps the weights.
    twin = draft.with_framework(PY)
    np.testing.assert_array_equal(
        logits_of(twin(SMALL)[0]), logits_of(fw.numpy(logits) if fw.is_tensor(logits) else logits)
    )


@pytest.mark.parametrize("agreement", [0.0, -0.5, 1.5])
def test_agreement_must_be_a_rate(agreement):
    with pytest.raises(ValueError, match="agreement"):
        draft_target_pair(PY, 0, agreement)
