"""Milestone 6 kit, ops: softplus, l2norm, causal_conv1d, partial_rope, gated_delta_layer,
eb_accept. One contract, four implementations, each checked against an independent
float64 computation written here."""

import importlib.util
import math

import numpy as np
import pytest
from conftest import REPO_ROOT

from learnai.judge import frameworks
from learnai.kit import GDNConfig, gdn_gates_ref, gdn_layer_ref, gdn_params, gdn_step_ref, ops

BACKENDS = ["numpy", "torch", "jax", "mlx"]


@pytest.fixture(params=BACKENDS)
def run(request):
    """Call an op on numpy inputs through one backend; get numpy back (also from tuples)."""
    name = request.param
    if name == "numpy":
        return lambda fn, *args, **kwargs: fn(*args, **kwargs)
    try:
        fw = frameworks.get(name)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))

    def call(fn, *args, **kwargs):
        out = fn(*fw.to_framework(list(args)), **kwargs)
        leaves = out if isinstance(out, tuple) else (out,)
        flat = [x for leaf in leaves for x in (leaf if isinstance(leaf, tuple) else (leaf,))]
        assert all(fw.is_tensor(x) for x in flat), f"{fn.__name__} must return tensors"
        return fw.to_numpy(out)

    return call


def randn(*shape, seed=0, scale=1.0):
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32)


# -- softplus, l2norm -----------------------------------------------------------------------


def test_softplus_is_accurate_where_the_result_is_tiny(run):
    x = np.array([-30.0, -20.0, -10.0, -5.0, -1.0, 0.0, 1.0, 10.0, 30.0, 200.0], dtype=np.float32)
    got = run(ops.softplus, x)
    want = np.log1p(np.exp(-np.abs(x.astype(np.float64)))) + np.maximum(x, 0)
    assert got.dtype == np.float32
    # Relative accuracy everywhere, including x = -10 (4.54e-5), where log(1 + e^x)
    # computed naively in float32 is off by 6e-4 relatively.
    np.testing.assert_allclose(got, want, rtol=2e-6)
    assert np.isfinite(got).all() and got[-1] == 200.0


def test_mlx_nn_softplus_is_the_inexact_one():
    # The fact behind the guide's note: mlx.nn.softplus loses the small results, while
    # ops.softplus and mx.logaddexp(x, 0) do not. If this starts failing, MLX fixed it.
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    x = mx.array([-10.0], dtype=mx.float32)
    exact = math.log1p(math.exp(-10.0))
    assert abs(nn.softplus(x).item() / exact - 1) > 3e-4
    assert abs(mx.logaddexp(x, mx.zeros_like(x)).item() / exact - 1) < 1e-6
    assert abs(ops.softplus(x).item() / exact - 1) < 1e-6


def test_l2norm_matches_the_shipped_formula(run):
    x = randn(3, 4, 6, scale=2.0)
    x[0, 0] = 0.0  # a zero vector stays zero: the eps is inside the root
    got = run(ops.l2norm, x)
    want = x * (1.0 / np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True) + 1e-6))
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(got[1:], axis=-1), 1.0, rtol=1e-5)
    assert not got[0, 0].any()
    np.testing.assert_allclose(
        run(ops.l2norm, x, eps=0.5), x / np.sqrt((x**2).sum(-1, keepdims=True) + 0.5), rtol=1e-5
    )


# -- causal_conv1d --------------------------------------------------------------------------


def conv_by_definition(x, w, state):
    """y[t, c] = sum_j w[c, j] * padded[t + j, c], in float64 loops."""
    kernel = w.shape[-1]
    padded = np.concatenate((state, x), axis=0).astype(np.float64)
    y = np.zeros(x.shape, dtype=np.float64)
    for t in range(x.shape[0]):
        for j in range(kernel):
            y[t] += w[:, j] * padded[t + j]
    return y, padded[padded.shape[0] - (kernel - 1) :]


@pytest.mark.parametrize("kernel", [1, 2, 4, 7])
def test_causal_conv1d_is_a_depthwise_cross_correlation_with_state(run, kernel):
    steps, channels = 9, 5
    x, w = randn(steps, channels), randn(channels, kernel, seed=1)
    state = randn(kernel - 1, channels, seed=2)
    y, new_state = run(ops.causal_conv1d, x, w, state)
    want_y, want_state = conv_by_definition(x, w, state)
    assert y.shape == (steps, channels) and new_state.shape == (kernel - 1, channels)
    np.testing.assert_allclose(y, want_y, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(new_state, want_state, rtol=1e-6)
    # No state means zeros, and the checkpoint layout (C, 1, K) is the same kernel.
    y0, _ = run(ops.causal_conv1d, x, w)
    np.testing.assert_allclose(
        y0, conv_by_definition(x, w, np.zeros((kernel - 1, channels), np.float32))[0],
        rtol=1e-5, atol=1e-6,
    )  # fmt: skip
    y3, _ = run(ops.causal_conv1d, x, w[:, None, :], state)
    np.testing.assert_array_equal(y3, y)


def test_causal_conv1d_agrees_with_torch_conv1d():
    torch = pytest.importorskip("torch")
    x, w = randn(11, 6), randn(6, 1, 4, seed=1)
    y, _ = ops.causal_conv1d(x, w)
    # The shipped layer: a grouped Conv1d padded by K - 1 on both sides, cut to T.
    want = torch.nn.functional.conv1d(
        torch.from_numpy(x.T[None]), torch.from_numpy(w), padding=3, groups=6
    )[0, :, :11].T.numpy()
    np.testing.assert_allclose(y, want, rtol=1e-5, atol=1e-6)


def test_causal_conv1d_any_chunking_gives_the_output_of_one_pass(run):
    x, w = randn(2, 12, 4), randn(4, 4, seed=1)  # with a batch axis
    whole, final = run(ops.causal_conv1d, x, w)
    for cuts in ([1, 2, 3, 12], [5, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
        state, pieces, start = None, [], 0
        for stop in cuts:
            piece, state = run(ops.causal_conv1d, x[:, start:stop], w, state)
            pieces.append(piece)
            start = stop
        np.testing.assert_allclose(np.concatenate(pieces, axis=1), whole, rtol=1e-5, atol=1e-6)
        # The state is the last K - 1 inputs, also when a piece is shorter than that.
        np.testing.assert_array_equal(state, x[:, -3:])
    np.testing.assert_array_equal(final, x[:, -3:])
    # Causal: output t does not depend on inputs after t.
    changed = x.copy()
    changed[:, 7:] += 1.0
    np.testing.assert_array_equal(run(ops.causal_conv1d, changed, w)[0][:, :7], whole[:, :7])


# -- partial_rope ---------------------------------------------------------------------------


def partial_rope_by_definition(x, positions, base, factor):
    dim = x.shape[-1]
    r = int(dim * factor)
    out = x.astype(np.float64).copy()
    for t, p in enumerate(positions):
        for i in range(r // 2):
            angle = float(p) * base ** (-2 * i / r)
            a, b = x[..., t, i].astype(np.float64), x[..., t, i + r // 2].astype(np.float64)
            out[..., t, i] = a * math.cos(angle) - b * math.sin(angle)
            out[..., t, i + r // 2] = a * math.sin(angle) + b * math.cos(angle)
    return out


@pytest.mark.parametrize(
    ("dim", "factor"), [(16, 0.25), (16, 0.5), (8, 1.0), (16, 0.0), (24, 0.75)]
)
def test_partial_rope_rotates_the_first_channels_half_split(run, dim, factor):
    x = randn(2, 3, 5, dim)
    positions = np.array([0, 1, 2, 7, 40])
    got = run(ops.partial_rope, x, positions, 10000.0, factor)
    assert got.dtype == np.float32 and got.shape == x.shape
    np.testing.assert_allclose(
        got, partial_rope_by_definition(x, positions, 10000.0, factor), rtol=1e-4, atol=1e-5
    )
    r = int(dim * factor)
    np.testing.assert_array_equal(got[..., r:], x[..., r:])  # the rest passes through untouched
    np.testing.assert_array_equal(got[..., 0, :], x[..., 0, :])  # position 0 is the identity
    # A rotation: the norm of the rotated block is kept.
    np.testing.assert_allclose(
        np.linalg.norm(got[..., :r], axis=-1), np.linalg.norm(x[..., :r], axis=-1), rtol=1e-5
    )


def test_partial_rope_scores_depend_on_position_differences_and_rows_of_positions(run):
    q, k = randn(1, 2, 6, 16), randn(1, 2, 6, 16, seed=1)
    q[:], k[:] = q[..., :1, :], k[..., :1, :]  # the same vector at every position
    positions = np.arange(6)
    rq = run(ops.partial_rope, q, positions, 100.0, 0.5)
    rk = run(ops.partial_rope, k, positions, 100.0, 0.5)
    scores = rq @ np.swapaxes(rk, -1, -2)
    for shift in range(1, 5):
        diagonal = np.diagonal(scores, shift, -2, -1)
        np.testing.assert_allclose(
            diagonal, np.broadcast_to(diagonal[..., :1], diagonal.shape), rtol=1e-4, atol=1e-5
        )
    # One row of positions per batch row, as ops.rope takes them.
    x = randn(2, 3, 4, 8)
    rows = np.array([[0, 1, 2, 3], [10, 11, 12, 13]])
    both = run(ops.partial_rope, x, rows, 10000.0, 0.5)
    for b in range(2):
        np.testing.assert_allclose(
            both[b], run(ops.partial_rope, x[b], rows[b], 10000.0, 0.5), rtol=1e-6, atol=1e-7
        )


def test_partial_rope_refuses_an_odd_rotated_width(run):
    with pytest.raises(ValueError, match="must be even"):
        run(ops.partial_rope, randn(3, 24), np.arange(3), 10000.0, 0.375)  # int(9.0) = 9


def test_partial_rope_is_not_ops_rope_but_a_permutation_of_it(run):
    # ops.rope pairs adjacent channels; partial_rope pairs channel i with i + r/2.
    x, positions = randn(4, 8), np.array([3, 4, 5, 6])
    half_split = run(ops.partial_rope, x, positions, 10000.0, 1.0)
    order = np.concatenate((np.arange(0, 8, 2), np.arange(1, 8, 2)))  # evens, then odds
    interleaved = run(ops.rope, x[:, np.argsort(order)], positions, 10000.0)
    np.testing.assert_allclose(half_split, interleaved[:, order], rtol=1e-5, atol=1e-6)
    assert not np.allclose(half_split, run(ops.rope, x, positions, 10000.0), atol=1e-3)


@pytest.mark.parametrize("name", ["torch", "jax", "mlx"])
def test_partial_rope_agrees_with_the_partial_rope_problem(name):
    """The shipped problem and the kit must be one function: same argument names
    (``base``), same half-split layout, same ``int(D * factor)``."""
    reference = REPO_ROOT / "problems" / "block-components" / "partial-rope" / "reference"
    if not (reference / f"{name}.py").exists():
        pytest.skip("the partial-rope problem is not in this checkout")
    try:
        fw = frameworks.get(name)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))
    spec = importlib.util.spec_from_file_location(f"partial_rope_{name}", reference / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dim, factor, base in [(16, 0.25, 1e7), (16, 0.5, 1e6), (8, 1.0, 1e4), (32, 0.375, 1e4)]:
        x = fw.tensor(randn(2, 7, dim, seed=dim))
        positions = fw.tensor(np.array([0, 1, 2, 3, 50, 51, 1000]))
        want = fw.numpy(module.partial_rope(x, positions, base=base, factor=factor))
        got = fw.numpy(ops.partial_rope(x, positions, base=base, factor=factor))
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
        # Across frameworks the float32 angle at position 1000 differs in its last bits.
        np.testing.assert_allclose(
            ops.partial_rope(fw.numpy(x), fw.numpy(positions), base, factor), want,
            rtol=1e-4, atol=2e-4,
        )  # fmt: skip


# -- gated_delta_layer ----------------------------------------------------------------------


def layer_by_definition(cfg, p, x, state=None):
    """The layer written out token by token in float64, from the spec's sentence."""
    p = {name: value.astype(np.float64) for name, value in p.items()}
    x = x.astype(np.float64)
    kernel, conv_dim = cfg.conv_kernel, cfg.conv_dim
    conv_state = np.zeros((kernel - 1, conv_dim)) if state is None else state[0].astype(np.float64)
    memory = (
        np.zeros((cfg.n_value_heads, cfg.value_head_dim, cfg.key_head_dim))
        if state is None
        else state[1].astype(np.float64)
    )
    silu = lambda z: z / (1 + np.exp(-z))
    repeat = cfg.n_value_heads // cfg.n_key_heads
    out = []
    for row in x:
        projected = p["in_proj_qkv.weight"] @ row
        window = np.concatenate((conv_state, projected[None]), axis=0)  # the last K inputs
        conv_state = window[1:]
        mixed = silu((window.T * p["conv1d.weight"][:, 0, :]).sum(-1))
        q = mixed[: cfg.key_dim].reshape(cfg.n_key_heads, cfg.key_head_dim)
        k = mixed[cfg.key_dim : 2 * cfg.key_dim].reshape(cfg.n_key_heads, cfg.key_head_dim)
        v = mixed[2 * cfg.key_dim :].reshape(cfg.n_value_heads, cfg.value_head_dim)
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * cfg.key_head_dim**-0.5
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        z = (p["in_proj_z.weight"] @ row).reshape(cfg.n_value_heads, cfg.value_head_dim)
        a, b = p["in_proj_a.weight"] @ row, p["in_proj_b.weight"] @ row
        g = np.exp(-np.exp(p["A_log"]) * np.log1p(np.exp(a + p["dt_bias"])))
        beta = 1 / (1 + np.exp(-b))
        ys = []
        for h in range(cfg.n_value_heads):
            kh, qh = k[h // repeat], q[h // repeat]  # key head j serves value heads j*r .. j*r+r-1
            memory[h] = g[h] * memory[h]
            memory[h] = memory[h] + beta[h] * np.outer(v[h] - memory[h] @ kh, kh)
            y = memory[h] @ qh
            y = y / np.sqrt((y * y).mean() + cfg.norm_eps) * p["norm.weight"] * silu(z[h])
            ys.append(y)
        out.append(p["out_proj.weight"] @ np.concatenate(ys))
    return np.stack(out), (conv_state, memory)


GDN_SHAPES = [
    GDNConfig(),  # 2 key heads, 4 value heads
    GDNConfig(d_model=16, n_key_heads=3, n_value_heads=3, key_head_dim=4, value_head_dim=6),
    GDNConfig(d_model=24, n_key_heads=1, n_value_heads=4, key_head_dim=6, value_head_dim=4,
              conv_kernel=2, norm_eps=1e-5),
]  # fmt: skip


@pytest.mark.parametrize("cfg", GDN_SHAPES, ids=["2k4v", "3k3v", "1k4v"])
def test_gated_delta_layer_matches_the_definition(run, cfg):
    params = gdn_params(cfg, seed=4)
    x = randn(10, cfg.d_model, seed=1)
    out, (conv_state, memory) = run(
        ops.gated_delta_layer, x, params, None, cfg.n_key_heads, cfg.norm_eps
    )
    want, (want_conv, want_memory) = layer_by_definition(cfg, params, x)
    assert out.dtype == np.float32 and out.shape == (10, cfg.d_model)
    assert conv_state.shape == (cfg.conv_kernel - 1, cfg.conv_dim)
    assert memory.shape == (cfg.n_value_heads, cfg.value_head_dim, cfg.key_head_dim)
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(conv_state, want_conv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(memory, want_memory, rtol=1e-4, atol=1e-5)
    # Continuing from a nonzero state.
    more = randn(4, cfg.d_model, seed=2)
    out2, _ = run(
        ops.gated_delta_layer, more, params, (conv_state, memory), cfg.n_key_heads, cfg.norm_eps
    )
    want2, _ = layer_by_definition(cfg, params, more, (want_conv, want_memory))
    np.testing.assert_allclose(out2, want2, rtol=1e-4, atol=1e-5)
    # The numpy oracle is the same function.
    ref, _ = gdn_layer_ref(cfg, params, x)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_gated_delta_layer_any_chunking_gives_the_output_of_one_pass(run):
    cfg = GDNConfig()
    params = gdn_params(cfg, seed=2)
    x = randn(2, 14, cfg.d_model, seed=3)  # batched
    whole, (conv_whole, memory_whole) = run(ops.gated_delta_layer, x, params, None, 2)
    assert whole.shape == (2, 14, cfg.d_model) and memory_whole.shape == (2, 4, 8, 8)
    for cuts in ([14], [1, 14], [3, 4, 9, 14], list(range(1, 15))):
        state, pieces, start = None, [], 0
        for stop in cuts:
            piece, state = run(ops.gated_delta_layer, x[:, start:stop], params, state, 2)
            pieces.append(piece)
            start = stop
        np.testing.assert_allclose(np.concatenate(pieces, axis=1), whole, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(state[1], memory_whole, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(state[0], conv_whole, rtol=1e-5, atol=1e-6)
    # Batch rows are independent, and an unbatched call is one of them.
    alone, _ = run(ops.gated_delta_layer, x[1], params, None, 2)
    np.testing.assert_allclose(alone, whole[1], rtol=1e-4, atol=1e-5)


def test_gated_delta_layer_every_piece_of_state_and_every_parameter_matters():
    cfg = GDNConfig()
    params = gdn_params(cfg, seed=0)
    x = randn(12, cfg.d_model, seed=5)
    whole, _ = gdn_layer_ref(cfg, params, x)
    _, (conv_state, memory) = gdn_layer_ref(cfg, params, x[:6])
    tail = whole[6:]

    def gap(state):
        return float(np.abs(gdn_layer_ref(cfg, params, x[6:], state)[0] - tail).max())

    assert gap((conv_state, memory)) < 1e-5
    # Neither piece can stand in for the other: dropping either changes the output.
    assert gap((np.zeros_like(conv_state), memory)) > 1e-2
    assert gap((conv_state, np.zeros_like(memory))) > 1e-2
    assert gap(None) > 1e-2
    # The mutants of the specs are visible on these parameters.
    for name in params:
        if name in ("A_log", "dt_bias", "norm.weight"):
            changed = dict(params) | {name: np.zeros_like(params[name])}
            assert np.abs(gdn_layer_ref(cfg, changed, x)[0] - whole).max() > 1e-3, name
    flipped = dict(params) | {"conv1d.weight": params["conv1d.weight"][:, :, ::-1].copy()}
    assert np.abs(gdn_layer_ref(cfg, flipped, x)[0] - whole).max() > 1e-2
    # Key heads are repeated, not tiled: with 2 key heads for 4 value heads the layer
    # with n_key_heads=4 (no sharing, different split) is another function.
    tiled, _ = ops.gated_delta_layer(x, params, None, 1)
    assert np.abs(tiled - whole).max() > 1e-2


def test_gdn_step_and_gates_refs():
    rng = np.random.default_rng(0)
    heads, d_k, d_v = 3, 5, 4
    memory = rng.standard_normal((heads, d_v, d_k)).astype(np.float32)
    k = rng.standard_normal((heads, d_k)).astype(np.float32)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q, v = randn(heads, d_k, seed=1), randn(heads, d_v, seed=2)
    # With g = 1, beta = 1, and a unit key, the memory returns v under k whatever it held.
    y, new = gdn_step_ref(memory, q, k, v, np.ones(heads), np.ones(heads))
    np.testing.assert_allclose(new @ k[:, :, None], v[:, :, None], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(y, (new @ q[:, :, None])[:, :, 0], rtol=1e-6)  # read after write
    # beta = 0 writes nothing; g scales the state first.
    _, kept = gdn_step_ref(memory, q, k, v, np.full(heads, 0.5), np.zeros(heads))
    np.testing.assert_allclose(kept, 0.5 * memory, rtol=1e-6)

    a, b = randn(7, heads, seed=3, scale=3.0), randn(7, heads, seed=4, scale=3.0)
    a_log, dt_bias = randn(heads, seed=5), randn(heads, seed=6)
    g, beta = gdn_gates_ref(a, b, a_log, dt_bias)
    want_g = np.exp(
        -np.exp(a_log.astype(np.float64)) * np.log1p(np.exp(a + dt_bias.astype(np.float64)))
    )
    np.testing.assert_allclose(g, want_g, rtol=1e-5)
    np.testing.assert_allclose(beta, 1 / (1 + np.exp(-b.astype(np.float64))), rtol=1e-5)
    assert (g > 0).all() and (g <= 1).all() and g.dtype == np.float32


def test_gdn_config_and_params():
    cfg = GDNConfig()
    assert (cfg.key_dim, cfg.value_dim, cfg.conv_dim) == (16, 32, 64)
    params = gdn_params(cfg, seed=3)
    assert {name: value.shape for name, value in params.items()} == {
        "in_proj_qkv.weight": (64, 32), "in_proj_z.weight": (32, 32),
        "in_proj_a.weight": (4, 32), "in_proj_b.weight": (4, 32),
        "conv1d.weight": (64, 1, 4), "A_log": (4,), "dt_bias": (4,),
        "norm.weight": (8,), "out_proj.weight": (32, 32),
    }  # fmt: skip
    assert all(value.dtype == np.float32 for value in params.values())
    again, other = gdn_params(cfg, seed=3), gdn_params(cfg, seed=4)
    assert all(np.array_equal(params[name], again[name]) for name in params)
    assert not np.array_equal(params["A_log"], other["A_log"])
    # Away from the initialization values, so each one matters.
    assert np.abs(params["dt_bias"] - 1).min() > 0 and np.abs(params["norm.weight"] - 1).min() > 0
    with pytest.raises(ValueError, match="multiple of n_key_heads"):
        GDNConfig(n_key_heads=3, n_value_heads=4)
    with pytest.raises(ValueError, match="must have shape"):
        gdn_layer_ref(cfg, params, np.zeros((3, 5), np.float32))


# -- eb_accept ------------------------------------------------------------------------------


def entropies(logits):
    z = logits.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    return -(p * np.log(np.where(p > 0, p, 1.0))).sum(-1)


def accept_by_definition(logits, bound):
    h = entropies(logits)
    mask = np.zeros(h.shape, dtype=bool)
    for row in np.ndindex(h.shape[:-1]):
        spent = 0.0
        for position in sorted(range(h.shape[-1]), key=lambda i: (h[row][i], i)):
            if spent > bound:
                break
            mask[row][position] = True
            spent += h[row][position]
    return mask


@pytest.mark.parametrize("bound", [0.0, 0.05, 0.5, 1.5, 4.0, 100.0])
def test_eb_accept_takes_low_entropy_positions_while_the_sum_before_fits(run, bound):
    logits = randn(3, 9, 12, scale=3.0) * np.linspace(0.2, 3.0, 9)[None, :, None]
    got = run(ops.eb_accept, logits.astype(np.float32), bound)
    assert got.dtype == np.bool_ and got.shape == (3, 9)
    np.testing.assert_array_equal(got, accept_by_definition(logits, bound))
    h = entropies(logits)
    # The lowest-entropy position is always accepted, even with a bound of zero.
    assert got[np.arange(3), h.argmin(-1)].all()
    # What is accepted is a prefix of the entropy order.
    for row in range(3):
        assert h[row][got[row]].max() <= h[row][~got[row]].min() if (~got[row]).any() else True
    if bound == 100.0:
        assert got.all()


def test_eb_accept_is_monotone_in_the_bound_and_handles_ties_and_minus_inf(run):
    logits = randn(8, 10, scale=2.0)
    previous = np.zeros(8, dtype=bool)
    for bound in (0.0, 0.3, 0.6, 1.0, 2.0, 5.0, 20.0):
        got = run(ops.eb_accept, logits, bound)
        assert (got | ~previous).all()  # a larger bound never accepts fewer positions
        previous = got
    # Equal entropies: the lowest index goes first.
    same = np.tile(np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32), (5, 1))
    h = float(entropies(same)[0])
    np.testing.assert_array_equal(
        run(ops.eb_accept, same, 1.5 * h), [True, True, False, False, False]
    )
    np.testing.assert_array_equal(run(ops.eb_accept, same, 0.0), [True, False, False, False, False])
    # A masked token (-inf) has probability 0 and contributes no entropy, not NaN.
    masked = logits.copy()
    masked[:, :3] = -np.inf
    got = run(ops.eb_accept, masked, 1.0)
    np.testing.assert_array_equal(got, accept_by_definition(masked, 1.0))


def test_new_ops_refuse_foreign_types():
    for fn, args in [
        (ops.softplus, ([1.0],)),
        (ops.l2norm, ([1.0],)),
        (ops.eb_accept, ([[1.0]], 1.0)),
    ]:
        with pytest.raises(TypeError, match="torch, JAX, MLX, or numpy"):
            fn(*args)
