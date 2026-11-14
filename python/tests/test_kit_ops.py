"""whetstone.kit.ops: one contract, four implementations."""

import numpy as np
import pytest

from whetstone.judge import frameworks
from whetstone.kit import ops

NEG_INF = float("-inf")


@pytest.fixture(params=["numpy", "torch", "jax", "mlx"])
def run(request):
    """Call an op on numpy inputs through one backend; get numpy back."""
    name = request.param
    if name == "numpy":
        return lambda fn, *args, **kwargs: fn(*args, **kwargs)
    try:
        fw = frameworks.get(name)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))

    def call(fn, *args, **kwargs):
        converted = [fw.tensor(a) if isinstance(a, np.ndarray) else a for a in args]
        out = fn(*converted, **kwargs)
        assert fw.is_tensor(out), f"{fn.__name__} must return a {fw.tensor_type_name}"
        return fw.numpy(out)

    return call


def rng(seed=0):
    return np.random.default_rng(seed)


def randn(*shape, seed=0, scale=1.0):
    return (rng(seed).standard_normal(shape) * scale).astype(np.float32)


def test_softmax_and_log_softmax(run):
    x = randn(3, 5, scale=3.0)
    x[0, 1] = NEG_INF
    x[1] = [1000.0, 1001.0, 1002.0, -1000.0, 0.0]
    for axis in (-1, 0):
        got = run(ops.softmax, x, axis=axis)
        e = np.exp(x.astype(np.float64) - x.max(axis=axis, keepdims=True))
        want = e / e.sum(axis=axis, keepdims=True)
        assert got.dtype == np.float32
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)
    assert run(ops.softmax, x)[0, 1] == 0.0
    logs = run(ops.log_softmax, x)
    assert logs.dtype == np.float32 and logs[0, 1] == NEG_INF
    np.testing.assert_allclose(logs[1], [-2.4076059, -1.4076059, -0.4076059, -2002.4076, -1002.4076],
                               rtol=1e-5)  # fmt: skip


def test_rms_norm(run):
    x, weight = randn(2, 3, 8, scale=4.0), randn(8, seed=1)
    got = run(ops.rms_norm, x, weight)
    want = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * weight
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # eps matters for tiny inputs.
    tiny = np.full((1, 4), 1e-4, dtype=np.float32)
    ones = np.ones(4, dtype=np.float32)
    np.testing.assert_allclose(run(ops.rms_norm, tiny, ones, eps=1e-2), tiny / 0.1, rtol=1e-4)


def rope_by_hand(x, positions, base):
    out = np.zeros_like(x, dtype=np.float64)
    dim = x.shape[-1]
    for t in range(x.shape[-2]):
        for i in range(dim // 2):
            angle = float(positions[t]) * base ** (-2 * i / dim)
            a, b = x[..., t, 2 * i].astype(np.float64), x[..., t, 2 * i + 1].astype(np.float64)
            out[..., t, 2 * i] = a * np.cos(angle) - b * np.sin(angle)
            out[..., t, 2 * i + 1] = a * np.sin(angle) + b * np.cos(angle)
    return out


def test_rope_with_shared_positions(run):
    x = randn(2, 3, 5, 8)
    positions = np.array([0, 1, 2, 7, 40])
    got = run(ops.rope, x, positions)
    assert got.dtype == np.float32 and got.shape == x.shape
    np.testing.assert_allclose(got, rope_by_hand(x, positions, 10000.0), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(got[..., 0, :], x[..., 0, :])  # position 0 is the identity
    other = run(ops.rope, x, positions, base=100.0)
    np.testing.assert_allclose(other, rope_by_hand(x, positions, 100.0), rtol=1e-4, atol=1e-5)
    # A 2-D x (T, D) works too.
    np.testing.assert_allclose(run(ops.rope, x[0, 0], positions), got[0, 0], rtol=1e-5, atol=1e-6)


def test_rope_with_one_row_of_positions_per_batch_row(run):
    x = randn(2, 3, 4, 6, seed=2)
    positions = np.array([[0, 1, 2, 3], [5, 6, 7, 8]])
    got = run(ops.rope, x, positions)
    for b in range(2):
        np.testing.assert_allclose(got[b], rope_by_hand(x[b], positions[b], 10000.0),
                                   rtol=1e-4, atol=1e-5)  # fmt: skip


def attention_by_hand(q, k, v, allowed):
    scores = (q @ np.swapaxes(k, -1, -2)).astype(np.float64) / np.sqrt(q.shape[-1])
    scores = np.where(allowed, scores, -np.inf)
    e = np.exp(scores - scores.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)) @ v


def test_attention_without_and_with_masks(run):
    q, k, v = randn(2, 4, 5, 8), randn(2, 4, 6, 8, seed=1), randn(2, 4, 6, 3, seed=2)
    everything = np.ones((5, 6), dtype=bool)
    got = run(ops.attention, q, k, v)
    assert got.dtype == np.float32 and got.shape == (2, 4, 5, 3)
    np.testing.assert_allclose(got, attention_by_hand(q, k, v, everything), rtol=1e-4, atol=1e-5)

    allowed = rng(3).random((2, 1, 5, 6)) < 0.6
    allowed[..., 0] = True  # every query sees at least one key
    want = attention_by_hand(q, k, v, allowed)
    np.testing.assert_allclose(run(ops.attention, q, k, v, allowed), want, rtol=1e-4, atol=1e-5)
    additive = np.where(allowed, 0.0, NEG_INF).astype(np.float32)
    np.testing.assert_allclose(run(ops.attention, q, k, v, additive), want, rtol=1e-4, atol=1e-5)


def test_attention_gives_zeros_to_a_query_that_sees_nothing(run):
    q, k, v = randn(1, 2, 3, 4), randn(1, 2, 3, 4, seed=1), randn(1, 2, 3, 4, seed=2)
    allowed = np.tril(np.ones((3, 3), dtype=bool))
    allowed[1] = False
    for mask in (allowed, np.where(allowed, 0.0, NEG_INF).astype(np.float32)):
        got = run(ops.attention, q, k, v, mask)
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got[:, :, 1], np.zeros((1, 2, 4), dtype=np.float32))
        np.testing.assert_allclose(got[:, :, 0], v[:, :, 0], rtol=1e-5, atol=1e-6)


def test_attention_groups_query_heads_over_fewer_kv_heads(run):
    q, k, v = randn(2, 6, 4, 8), randn(2, 2, 5, 8, seed=1), randn(2, 2, 5, 8, seed=2)
    got = run(ops.attention, q, k, v)
    # Query head h uses KV head h // 3: heads 0-2 share KV head 0, heads 3-5 share head 1.
    want = attention_by_hand(q, np.repeat(k, 3, axis=1), np.repeat(v, 3, axis=1), True)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    wrong = attention_by_hand(q, np.tile(k, (1, 3, 1, 1)), np.tile(v, (1, 3, 1, 1)), True)
    assert not np.allclose(got, wrong, atol=1e-3)
    one = run(ops.attention, q, k[:, :1], v[:, :1])  # a single KV head serves all
    np.testing.assert_allclose(
        one, attention_by_hand(q, np.repeat(k[:, :1], 6, 1), np.repeat(v[:, :1], 6, 1), True),
        rtol=1e-4, atol=1e-5,
    )  # fmt: skip
    with pytest.raises(ValueError, match="cannot share"):
        run(ops.attention, q, randn(2, 4, 5, 8), randn(2, 4, 5, 8))


def test_swiglu_with_non_square_weights(run):
    x = randn(2, 3, 6)
    w_gate, w_up, w_down = randn(10, 6, seed=1), randn(10, 6, seed=2), randn(4, 10, seed=3)
    got = run(ops.swiglu, x, w_gate, w_up, w_down)
    gate = x.astype(np.float64) @ w_gate.T
    want = (gate / (1 + np.exp(-gate)) * (x @ w_up.T)) @ w_down.T
    assert got.dtype == np.float32 and got.shape == (2, 3, 4)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    # Very negative gate inputs give zero, not NaN.
    big = np.full((1, 6), -1e4, dtype=np.float32)
    assert np.isfinite(run(ops.swiglu, big, np.abs(w_gate), w_up, w_down)).all()


def test_top_k_filter(run):
    logits = np.array([[1.0, 3.0, 3.0, 2.0, 3.0], [0.5, NEG_INF, 0.1, NEG_INF, NEG_INF]],
                      dtype=np.float32)  # fmt: skip
    got = run(ops.top_k_filter, logits, 2)
    assert got.dtype == np.float32
    # Three logits tie at the top; the two with the lowest index survive.
    np.testing.assert_array_equal(got[0], [NEG_INF, 3.0, 3.0, NEG_INF, NEG_INF])
    np.testing.assert_array_equal(got[1], logits[1])
    # Fewer finite entries than k: the masked ones stay masked.
    np.testing.assert_array_equal(run(ops.top_k_filter, logits, 4)[1], logits[1])
    for no_op in (0, -1, 5, 9):
        np.testing.assert_array_equal(run(ops.top_k_filter, logits, no_op), logits)
    np.testing.assert_array_equal(
        run(ops.top_k_filter, logits, 1), [[NEG_INF, 3.0, NEG_INF, NEG_INF, NEG_INF],
                                           [0.5, NEG_INF, NEG_INF, NEG_INF, NEG_INF]],
    )  # fmt: skip
    wide = randn(2, 3, 16, seed=4)
    kept = np.isfinite(run(ops.top_k_filter, wide, 5))
    assert (kept.sum(-1) == 5).all()
    assert (np.where(kept, wide, np.inf).min(-1) >= np.where(kept, -np.inf, wide).max(-1)).all()


def top_p_by_hand(logits, p):
    out = np.full_like(logits, NEG_INF)
    for index in np.ndindex(logits.shape[:-1]):
        row = logits[index].astype(np.float64)
        probs = np.exp(row - row.max())
        probs /= probs.sum()
        mass = 0.0
        for token in sorted(range(len(row)), key=lambda t: (-probs[t], t)):
            if mass < p:
                out[index + (token,)] = logits[index + (token,)]
            mass += probs[token]
    return out


def test_top_p_filter(run):
    # Probabilities 0.4, 0.3, 0.2, 0.1.
    logits = np.log(np.array([[0.1, 0.4, 0.2, 0.3]], dtype=np.float32))
    kept = lambda p: np.isfinite(run(ops.top_p_filter, logits, p))[0].tolist()
    assert kept(0.05) == [False, True, False, False]  # the top token always survives
    assert kept(0.5) == [False, True, False, True]  # 0.4 < 0.5: the token crossing p is kept
    assert kept(0.75) == [False, True, True, True]
    assert kept(1.0) == [True, True, True, True]
    got = run(ops.top_p_filter, logits, 0.5)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got[0, [1, 3]], logits[0, [1, 3]])  # kept logits are untouched

    ties = np.zeros((1, 4), dtype=np.float32)  # four equal tokens: rank by index
    np.testing.assert_array_equal(np.isfinite(run(ops.top_p_filter, ties, 0.4)),
                                  [[True, True, False, False]])  # fmt: skip
    masked = np.array([[0.0, NEG_INF, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(run(ops.top_p_filter, masked, 1.0), masked)

    wide = randn(3, 2, 12, seed=5, scale=2.0)
    for p in (0.3, 0.9):
        np.testing.assert_array_equal(run(ops.top_p_filter, wide, p), top_p_by_hand(wide, p))


def test_min_p_filter(run):
    logits = np.log(np.array([[0.5, 0.26, 0.2, 0.04], [0.25, 0.25, 0.25, 0.25]], dtype=np.float32))
    got = run(ops.min_p_filter, logits, 0.5)
    assert got.dtype == np.float32
    # Threshold 0.5 * 0.5 = 0.25 relative to the top probability, not an absolute 0.5.
    np.testing.assert_array_equal(np.isfinite(got), [[True, True, False, False], [True] * 4])
    np.testing.assert_array_equal(got[0, :2], logits[0, :2])
    np.testing.assert_array_equal(run(ops.min_p_filter, logits, 0.0), logits)
    only_top = np.isfinite(run(ops.min_p_filter, logits, 1.0))
    np.testing.assert_array_equal(only_top, [[True, False, False, False], [True] * 4])
    masked = np.array([[0.0, NEG_INF, -5.0]], dtype=np.float32)
    np.testing.assert_array_equal(np.isfinite(run(ops.min_p_filter, masked, 0.1)),
                                  [[True, False, False]])  # fmt: skip


def test_filters_chain_and_softmax_renormalizes(run):
    logits = randn(4, 20, seed=6, scale=2.0)
    chained = run(ops.min_p_filter, run(ops.top_p_filter, run(ops.top_k_filter, logits, 10), 0.9),
                  0.05)  # fmt: skip
    probs = run(ops.softmax, chained)
    np.testing.assert_allclose(probs.sum(-1), np.ones(4), rtol=1e-5)
    assert ((probs > 0) == np.isfinite(chained)).all()
    assert (np.isfinite(chained).sum(-1) >= 1).all()


def test_ops_reject_values_that_are_not_tensors():
    with pytest.raises(TypeError, match="torch, JAX, MLX, or numpy"):
        ops.softmax([1.0, 2.0])


def test_jax_ops_can_be_jitted():
    try:
        fw = frameworks.get("jax")
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))
    import jax

    logits = randn(2, 9, seed=7)
    fn = jax.jit(lambda x: ops.min_p_filter(ops.top_p_filter(ops.top_k_filter(x, 4), 0.8), 0.1))
    want = ops.min_p_filter(ops.top_p_filter(ops.top_k_filter(logits, 4), 0.8), 0.1)
    np.testing.assert_array_equal(np.asarray(fn(fw.tensor(logits))), want)
    q = randn(1, 4, 3, 4)
    attend = jax.jit(lambda q, k, v: ops.attention(ops.rope(q, jax.numpy.arange(3)), k, v))
    out = attend(fw.tensor(q), fw.tensor(q[:, :2]), fw.tensor(q[:, :2]))
    assert out.shape == (1, 4, 3, 4)


def test_torch_ops_do_not_use_the_built_ins_a_problem_may_forbid(monkeypatch):
    import torch

    def boom(*args, **kwargs):
        raise AssertionError("ops must not call this built-in")

    for owner, name in [
        (torch, "softmax"), (torch, "log_softmax"), (torch, "logsumexp"), (torch, "topk"),
        (torch.nn.functional, "softmax"), (torch.nn.functional, "log_softmax"),
        (torch.nn.functional, "silu"), (torch.nn.functional, "scaled_dot_product_attention"),
        (torch.nn.functional, "rms_norm"), (torch.Tensor, "softmax"), (torch.Tensor, "topk"),
    ]:  # fmt: skip
        monkeypatch.setattr(owner, name, boom)
    x = torch.tensor(randn(2, 4, 3, 8))
    ops.softmax(x), ops.log_softmax(x), ops.rms_norm(x, x[0, 0, 0])
    ops.attention(ops.rope(x, torch.arange(3)), x[:, :2], x[:, :2])
    ops.swiglu(x, x[0, 0, :, :], x[0, 0, :, :], x[0, 0, :, :3].T)
    ops.min_p_filter(ops.top_p_filter(ops.top_k_filter(x, 3), 0.9), 0.1)
