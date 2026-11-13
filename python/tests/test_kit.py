import numpy as np
import pytest

from whetstone.judge import frameworks
from whetstone.kit import Counters, KVCache, TinyLM

PROMPT = [0, 5, 9, 33, 2, 7]


def logits_of(fw, value):
    return np.asarray(fw.numpy(value) if fw.is_tensor(value) else value, dtype=np.float32)


def test_speaks_the_solutions_framework(fw):
    lm = TinyLM(fw, seed=1)
    logits, cache = lm(fw.tensor(np.array(PROMPT)))
    if fw.name == "python":
        assert isinstance(logits, list)
    else:
        assert fw.is_tensor(logits)
    array = logits_of(fw, logits)
    assert array.shape == (len(PROMPT), lm.vocab_size)
    assert array.dtype == np.float32
    assert isinstance(cache, KVCache) and cache.length == len(PROMPT)
    # A plain list of token ids is accepted in every framework.
    from_list, _ = lm(PROMPT)
    np.testing.assert_array_equal(logits_of(fw, from_list), array)


def test_logits_are_identical_across_frameworks():
    reference = None
    for name in ("python", "torch", "jax", "mlx"):
        try:
            fw = frameworks.get(name)
        except frameworks.FrameworkUnavailable:
            continue
        got = logits_of(fw, TinyLM(fw, seed=4)(PROMPT)[0])
        if reference is None:
            reference = got
        np.testing.assert_array_equal(got, reference)


def test_cached_decoding_matches_a_full_forward_pass():
    fw = frameworks.get("python")
    lm = TinyLM(fw, seed=2)
    full = logits_of(fw, lm(PROMPT)[0])
    cache, rows = None, []
    for token in PROMPT:
        logits, cache = lm([token], cache)
        rows.append(logits_of(fw, logits)[0])
    np.testing.assert_allclose(np.stack(rows), full, rtol=1e-4, atol=1e-4)
    # Prefix in one call, remainder in another.
    _, cache = lm(PROMPT[:4])
    tail, cache = lm(PROMPT[4:], cache)
    np.testing.assert_allclose(logits_of(fw, tail), full[4:], rtol=1e-4, atol=1e-4)
    assert cache.length == len(PROMPT)


def test_it_is_causal():
    fw = frameworks.get("python")
    lm = TinyLM(fw, seed=2)
    a = logits_of(fw, lm([0, 5, 9, 33])[0])
    b = logits_of(fw, lm([0, 5, 9, 60])[0])
    np.testing.assert_array_equal(a[:3], b[:3])  # earlier positions cannot see the change
    assert not np.allclose(a[3], b[3])


def test_cache_is_immutable_and_can_branch():
    fw = frameworks.get("python")
    lm = TinyLM(fw, seed=5)
    _, cache = lm(PROMPT)
    first, after_first = lm([11], cache)
    other, _ = lm([12], cache)  # reuse the old cache: a second branch
    again, _ = lm([11], cache)
    assert cache.length == len(PROMPT) and after_first.length == len(PROMPT) + 1
    np.testing.assert_array_equal(logits_of(fw, first), logits_of(fw, again))
    assert not np.allclose(logits_of(fw, first), logits_of(fw, other))
    with pytest.raises(ValueError):
        cache.keys[0][...] = 0.0


def test_batches():
    fw = frameworks.get("python")
    lm = TinyLM(fw, seed=6)
    batch = np.array([PROMPT, PROMPT[::-1]])
    logits, cache = lm(batch)
    array = logits_of(fw, logits)
    assert array.shape == (2, len(PROMPT), lm.vocab_size)
    np.testing.assert_allclose(array[0], logits_of(fw, lm(PROMPT)[0]), rtol=1e-4, atol=1e-4)
    step, _ = lm(np.array([[3], [4]]), cache)
    assert logits_of(fw, step).shape == (2, 1, lm.vocab_size)
    with pytest.raises(ValueError, match="batch size"):
        lm(np.array([[3]]), cache)


def test_counters_measure_work():
    fw = frameworks.get("python")
    lm = TinyLM(fw, seed=7)
    assert lm.counters.snapshot() == {"forward_calls": 0, "tokens_processed": 0}
    _, cache = lm(PROMPT)
    lm([1], cache)
    lm(np.array([PROMPT, PROMPT]))
    assert lm.counters.snapshot() == {"forward_calls": 3, "tokens_processed": 6 + 1 + 12}
    assert lm.counters.tokens_processed == 19
    lm.counters.reset()
    assert lm.counters.forward_calls == 0
    assert repr(Counters("a")) == "Counters(a=0)"


def test_seeds_select_models_and_twins_share_weights():
    python, torch = frameworks.get("python"), frameworks.get("torch")
    a, b, c = TinyLM(python, seed=1), TinyLM(python, seed=1), TinyLM(python, seed=2)
    assert logits_of(python, a(PROMPT)[0]).tolist() == logits_of(python, b(PROMPT)[0]).tolist()
    assert logits_of(python, a(PROMPT)[0]).tolist() != logits_of(python, c(PROMPT)[0]).tolist()
    twin = a.with_framework(torch)
    assert twin.fw is torch and twin.counters.forward_calls == 0
    np.testing.assert_array_equal(torch.numpy(twin(PROMPT)[0]), logits_of(python, a(PROMPT)[0]))
    # The twin has its own counters: a was called three times, the twin once.
    assert twin.counters.forward_calls == 1 and a.counters.forward_calls == 3


def test_the_distribution_is_usable_for_decoding_problems():
    fw = frameworks.get("python")
    for seed in range(5):
        lm = TinyLM(fw, seed=seed)
        logits, cache = lm([0])
        tokens = []
        for _ in range(30):
            token = int(np.argmax(logits_of(fw, logits)[-1]))
            tokens.append(token)
            logits, cache = lm([token], cache)
        assert len(set(tokens)) >= 5  # not stuck on one token
        assert all(0 <= t < lm.vocab_size for t in tokens)


def test_invalid_inputs_are_rejected_with_clear_errors():
    fw = frameworks.get("python")
    lm = TinyLM(fw, seed=0, max_context=8)
    with pytest.raises(ValueError, match=r"token ids must be in \[0, 64\)"):
        lm([0, 64])
    with pytest.raises(ValueError, match="exceeds max_context 8"):
        lm(list(range(9)))
    _, cache = lm(list(range(6)))
    with pytest.raises(ValueError, match="exceeds max_context 8"):
        lm([1, 2, 3], cache)
    with pytest.raises(ValueError, match="must be integers"):
        lm([0.5, 1.5])
    with pytest.raises(ValueError, match="T >= 1"):
        lm([])
    with pytest.raises(ValueError):
        TinyLM(fw, d_model=30, n_heads=4)
