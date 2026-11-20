"""Milestone 6 kit, models: HybridConfig and its oracle, CountingKV, TinyDSA, the MTP
chain, BlockDenoiser. Tensor-facing objects run through every installed framework."""

import math

import numpy as np
import pytest

from learnai.judge import frameworks
from learnai.kit import (
    BlockDenoiser,
    CountingKV,
    DecoderConfig,
    DenoiserCache,
    HybridConfig,
    TinyDSA,
    decoder_forward_ref,
    decoder_hidden_ref,
    decoder_params,
    hybrid_forward_ref,
    hybrid_params,
    index_types,
    layer_params,
    mtp_draft_ref,
    mtp_layer_params,
    mtp_params,
    ops,
)

NEG_INF = float("-inf")
LINEAR, FULL = "linear_attention", "full_attention"


@pytest.fixture(params=["torch", "jax", "mlx"])
def fw(request):
    """The tensor frameworks: these objects hand out tensors."""
    try:
        return frameworks.get(request.param)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))


def randn(*shape, seed=0, scale=1.0):
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32)


# -- HybridConfig, hybrid_params ------------------------------------------------------------


def test_hybrid_config_schedule_and_derived_sizes():
    cfg = HybridConfig()
    assert cfg.layers == (LINEAR, LINEAR, LINEAR, FULL)
    assert HybridConfig(n_layers=7, full_attention_interval=3).layers == (
        LINEAR, LINEAR, FULL, LINEAR, LINEAR, FULL, LINEAR,
    )  # fmt: skip
    # Qwen3.5-0.8B: 24 layers, full attention at 3, 7, ..., 23.
    real = HybridConfig(n_layers=24).layers
    assert [i for i, kind in enumerate(real) if kind == FULL] == [3, 7, 11, 15, 19, 23]
    # An explicit list wins over the interval, and a list is stored as a tuple.
    explicit = HybridConfig(n_layers=3, layer_types=[FULL, LINEAR, FULL])
    assert explicit.layers == (FULL, LINEAR, FULL) and hash(explicit) == hash(explicit)
    assert cfg.rotary_dim == 4 and HybridConfig(partial_rotary_factor=0.5).rotary_dim == 8
    gdn = cfg.gdn
    assert (gdn.d_model, gdn.n_key_heads, gdn.n_value_heads, gdn.conv_kernel) == (32, 2, 4, 4)
    assert (gdn.key_dim, gdn.value_dim, gdn.norm_eps) == (16, 32, 1e-6)
    for bad, match in [
        ({"n_heads": 3}, "multiple of n_kv_heads"),
        ({"head_dim": 12}, "must be even"),  # int(12 * 0.25) = 3
        ({"layer_types": (LINEAR,)}, "one entry per layer"),
        ({"n_layers": 1, "layer_types": ("attention",)}, "must be"),
        ({"n_key_heads": 3}, "multiple of n_key_heads"),
        ({"d_mlp": 0}, "at least 1"),
    ]:
        with pytest.raises(ValueError, match=match):
            HybridConfig(**bad)


def test_hybrid_params_are_named_like_the_checkpoint():
    cfg = HybridConfig()
    params = hybrid_params(cfg, seed=1)
    shapes = {name: value.shape for name, value in params.items()}
    assert (
        next(iter(shapes)) == "model.embed_tokens.weight"
        and list(shapes)[-1] == "model.norm.weight"
    )
    assert "lm_head.weight" not in shapes and shapes["model.embed_tokens.weight"] == (64, 32)
    assert {name: shape for name, shape in shapes.items() if ".layers.0." in name} == {
        "model.layers.0.input_layernorm.weight": (32,),
        "model.layers.0.linear_attn.in_proj_qkv.weight": (64, 32),
        "model.layers.0.linear_attn.in_proj_z.weight": (32, 32),
        "model.layers.0.linear_attn.in_proj_a.weight": (4, 32),
        "model.layers.0.linear_attn.in_proj_b.weight": (4, 32),
        "model.layers.0.linear_attn.conv1d.weight": (64, 1, 4),
        "model.layers.0.linear_attn.A_log": (4,),
        "model.layers.0.linear_attn.dt_bias": (4,),
        "model.layers.0.linear_attn.norm.weight": (8,),
        "model.layers.0.linear_attn.out_proj.weight": (32, 32),
        "model.layers.0.post_attention_layernorm.weight": (32,),
        "model.layers.0.mlp.gate_proj.weight": (48, 32),
        "model.layers.0.mlp.up_proj.weight": (48, 32),
        "model.layers.0.mlp.down_proj.weight": (32, 48),
    }
    assert {name: shape for name, shape in shapes.items() if ".layers.3.self_attn" in name} == {
        # q_proj carries the query and the gate: 2 * head_dim per head.
        "model.layers.3.self_attn.q_proj.weight": (4 * 2 * 16, 32),
        "model.layers.3.self_attn.k_proj.weight": (2 * 16, 32),
        "model.layers.3.self_attn.v_proj.weight": (2 * 16, 32),
        "model.layers.3.self_attn.o_proj.weight": (32, 4 * 16),
        "model.layers.3.self_attn.q_norm.weight": (16,),
        "model.layers.3.self_attn.k_norm.weight": (16,),
    }
    assert not any("self_attn" in name for name in shapes if ".layers.0." in name)
    assert all(value.dtype == np.float32 for value in params.values())
    again = hybrid_params(cfg, seed=1)
    assert all(np.array_equal(params[name], again[name]) for name in params)
    assert not np.array_equal(
        params["model.norm.weight"], hybrid_params(cfg, 2)["model.norm.weight"]
    )
    untied = hybrid_params(HybridConfig(tie_embeddings=False), seed=1)
    assert untied["lm_head.weight"].shape == (64, 32)
    assert set(layer_params(params, 3)) >= {"self_attn.q_norm.weight", "mlp.up_proj.weight"}


# -- hybrid_forward_ref ---------------------------------------------------------------------

HYBRIDS = {
    "default": HybridConfig(),
    "interval2": HybridConfig(n_layers=5, full_attention_interval=2, partial_rotary_factor=0.5),
    "mixed": HybridConfig(
        n_layers=3, layer_types=(FULL, LINEAR, FULL), n_heads=2, n_kv_heads=1, head_dim=8,
        n_key_heads=2, n_value_heads=2, conv_kernel=3, tie_embeddings=False,
    ),
    "all_linear": HybridConfig(n_layers=2, layer_types=(LINEAR, LINEAR)),
}  # fmt: skip


@pytest.mark.parametrize("name", HYBRIDS)
def test_hybrid_prefill_in_chunks_then_token_by_token_equals_one_pass(name):
    cfg = HYBRIDS[name]
    params = hybrid_params(cfg, seed=3)
    tokens = np.random.default_rng(1).integers(0, cfg.vocab_size, 17)
    whole, final = hybrid_forward_ref(cfg, params, tokens)
    assert whole.shape == (17, cfg.vocab_size) and whole.dtype == np.float32
    assert 1.0 < whole.std() < 4.0  # far above float32 rounding
    for cuts in ([5, 11, 12, 13, 14, 15, 16, 17], [1, 17], list(range(1, 18))):
        state, pieces, start = None, [], 0
        for stop in cuts:
            piece, state = hybrid_forward_ref(cfg, params, tokens[start:stop], state)
            pieces.append(piece)
            start = stop
        np.testing.assert_allclose(np.concatenate(pieces), whole, rtol=1e-3, atol=1e-4)
        for kept, expected in zip(state, final, strict=True):
            for a, b in zip(kept, expected, strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    # The state: (conv_state, S) for a recurrent layer, (k, v) for an attention layer.
    for kind, (first, second) in zip(cfg.layers, final, strict=True):
        if kind == LINEAR:
            assert first.shape == (cfg.conv_kernel - 1, cfg.gdn.conv_dim)
            assert second.shape == (cfg.n_value_heads, cfg.value_head_dim, cfg.key_head_dim)
        else:
            assert first.shape == second.shape == (cfg.n_kv_heads, 17, cfg.head_dim)
    # Causal: a row of logits depends on the tokens up to it only.
    np.testing.assert_allclose(
        hybrid_forward_ref(cfg, params, tokens[:9])[0], whole[:9], rtol=1e-3, atol=1e-4
    )


def test_hybrid_forward_handles_batches_and_does_not_modify_the_state():
    cfg = HybridConfig()
    params = hybrid_params(cfg, seed=0)
    tokens = np.random.default_rng(2).integers(0, cfg.vocab_size, (3, 9))
    logits, state = hybrid_forward_ref(cfg, params, tokens[:, :6])
    assert logits.shape == (3, 6, 64)
    assert state[0][1].shape == (3, 4, 8, 8) and state[3][0].shape == (3, 2, 6, 16)
    snapshot = [tuple(part.copy() for part in layer) for layer in state]
    more, longer = hybrid_forward_ref(cfg, params, tokens[:, 6:], state)
    for kept, before in zip(state, snapshot, strict=True):
        for a, b in zip(kept, before, strict=True):
            np.testing.assert_array_equal(a, b)
    assert longer[3][0].shape == (3, 2, 9, 16)
    for row in range(3):
        alone, _ = hybrid_forward_ref(cfg, params, tokens[row])
        np.testing.assert_allclose(logits[row], alone[:6], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(more[row], alone[6:], rtol=1e-3, atol=1e-4)
    for bad, match in [
        (np.zeros((0,), np.int64), "T >= 1"),
        (np.array([64]), "token ids"),
        (np.array([0.5]), "integers"),
    ]:
        with pytest.raises(ValueError, match=match):
            hybrid_forward_ref(cfg, params, bad)
    with pytest.raises(ValueError, match="one entry per layer"):
        hybrid_forward_ref(cfg, params, tokens[0], state[:2])


def test_hybrid_forward_is_sensitive_to_what_the_spec_mutants_break():
    cfg = HybridConfig(n_layers=4)
    params = hybrid_params(cfg, seed=5)
    tokens = np.random.default_rng(3).integers(0, cfg.vocab_size, 12)
    whole, _ = hybrid_forward_ref(cfg, params, tokens)
    _head, state = hybrid_forward_ref(cfg, params, tokens[:7])

    def tail(changed_state):
        return np.abs(
            hybrid_forward_ref(cfg, params, tokens[7:], changed_state)[0] - whole[7:]
        ).max()

    assert tail(state) < 1e-4
    # A recurrent layer whose state is not carried restarts from zeros.
    assert tail([None, *state[1:]]) > 1e-2
    # RoPE positions that restart at 0: drop the cached keys' count by handing the
    # attention layer an empty past of the wrong length is not expressible, so compare
    # with a run whose attention layer forgot its cache instead.
    assert tail([*state[:3], None]) > 1e-2
    # The other schedule (i % interval == 0) is a different model.
    other = HybridConfig(n_layers=4, layer_types=(FULL, LINEAR, LINEAR, LINEAR))
    assert other.layers != cfg.layers
    # The partial rotation matters: another factor, other logits.
    wider = HybridConfig(n_layers=4, partial_rotary_factor=0.5)
    assert np.abs(hybrid_forward_ref(wider, params, tokens)[0] - whole).max() > 1e-2
    # So do the gate half of q_proj and the QK norms.
    name = "model.layers.3.self_attn.q_proj.weight"
    rows = params[name].reshape(cfg.n_heads, 2, cfg.head_dim, cfg.d_model).copy()
    rows[:, 1] = 0.0  # every gate becomes sigmoid(0) = 0.5: not the same as no gate either
    ungated = dict(params) | {name: rows.reshape(params[name].shape)}
    assert np.abs(hybrid_forward_ref(cfg, ungated, tokens)[0] - whole).max() > 1e-2
    for norm in ("q_norm", "k_norm"):
        name = f"model.layers.3.self_attn.{norm}.weight"
        plain = dict(params) | {name: np.ones_like(params[name])}
        assert np.abs(hybrid_forward_ref(cfg, plain, tokens)[0] - whole).max() > 1e-3, norm


def hybrid_forward_with(fw, cfg, params, tokens, state=None):
    """The forward pass a reference solution would write: kit ops and the framework's
    own elementary functions, the same code for torch, JAX, and MLX."""
    xp = fw.xp
    ids = fw.tensor(np.asarray(tokens))
    new = int(ids.shape[0])
    state = [None] * cfg.n_layers if state is None else state
    seen = next(
        (int(s[0].shape[-2]) for s, kind in zip(state, cfg.layers, strict=True)
         if kind == FULL and s is not None), 0,
    )  # fmt: skip
    positions = fw.tensor(np.arange(seen, seen + new))
    mask = fw.tensor(np.arange(seen + new)[None, :] <= np.arange(seen, seen + new)[:, None])
    x = params["model.embed_tokens.weight"][ids]
    out_state = []
    for i, kind in enumerate(cfg.layers):
        block = layer_params(params, i)
        h = ops.rms_norm(x, block["input_layernorm.weight"], cfg.norm_eps)
        if kind == LINEAR:
            linear = {k[len("linear_attn."):]: v for k, v in block.items() if k.startswith("linear_attn.")}  # fmt: skip
            mixed, kept = ops.gated_delta_layer(h, linear, state[i], cfg.n_key_heads, cfg.norm_eps)
        else:
            both = (h @ block["self_attn.q_proj.weight"].T).reshape(
                new, cfg.n_heads, 2 * cfg.head_dim
            )
            q, gate = both[..., : cfg.head_dim], both[..., cfg.head_dim :]
            k = (h @ block["self_attn.k_proj.weight"].T).reshape(new, cfg.n_kv_heads, cfg.head_dim)
            v = (h @ block["self_attn.v_proj.weight"].T).reshape(new, cfg.n_kv_heads, cfg.head_dim)
            q = ops.rms_norm(q, block["self_attn.q_norm.weight"], cfg.norm_eps)
            k = ops.rms_norm(k, block["self_attn.k_norm.weight"], cfg.norm_eps)
            q, k, v = (xp.swapaxes(t, 0, 1) for t in (q, k, v))
            q = ops.partial_rope(q, positions, cfg.rope_base, cfg.partial_rotary_factor)
            k = ops.partial_rope(k, positions, cfg.rope_base, cfg.partial_rotary_factor)
            if state[i] is not None:
                k = xp.concatenate((state[i][0], k), -2)
                v = xp.concatenate((state[i][1], v), -2)
            attended = xp.swapaxes(ops.attention(q, k, v, mask), 0, 1)
            width = cfg.n_heads * cfg.head_dim
            gate = 1.0 / (1.0 + xp.exp(-gate.reshape(new, width)))
            mixed = (attended.reshape(new, width) * gate) @ block["self_attn.o_proj.weight"].T
            kept = (k, v)
        out_state.append(kept)
        x = x + mixed
        h = ops.rms_norm(x, block["post_attention_layernorm.weight"], cfg.norm_eps)
        x = x + ops.swiglu(
            h, block["mlp.gate_proj.weight"], block["mlp.up_proj.weight"], block["mlp.down_proj.weight"]
        )  # fmt: skip
    x = ops.rms_norm(x, params["model.norm.weight"], cfg.norm_eps)
    return x @ params["model.embed_tokens.weight"].T, out_state


def test_hybrid_forward_can_be_written_from_kit_ops_in_every_framework(fw):
    cfg = HybridConfig()
    assert fw.to_framework(cfg) is cfg  # a frozen dataclass passes through ctx.call unchanged
    params_np = hybrid_params(cfg, seed=7)
    params = fw.to_framework(params_np)
    tokens = np.random.default_rng(4).integers(0, cfg.vocab_size, 13)
    want, want_state = hybrid_forward_ref(cfg, params_np, tokens)
    got, state = hybrid_forward_with(fw, cfg, params, tokens)
    assert fw.is_tensor(got)
    np.testing.assert_allclose(fw.numpy(got), want, rtol=1e-3, atol=1e-4)
    # Prefill, then one token at a time, in the framework, against the one-pass oracle.
    pieces, state, start = [], None, 0
    for stop in (6, 10, 11, 12, 13):
        piece, state = hybrid_forward_with(fw, cfg, params, tokens[start:stop], state)
        pieces.append(fw.numpy(piece))
        start = stop
    np.testing.assert_allclose(np.concatenate(pieces), want, rtol=1e-3, atol=1e-4)
    for kept, expected in zip(fw.to_numpy(state), want_state, strict=True):
        for a, b in zip(kept, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    # The oracle accepts the state a solution returns (tensors converted to numpy).
    _, prefix_state = hybrid_forward_with(fw, cfg, params, tokens[:10])
    rest, _ = hybrid_forward_ref(cfg, params_np, tokens[10:], fw.to_numpy(prefix_state))
    np.testing.assert_allclose(rest, want[10:], rtol=1e-3, atol=1e-4)


# -- CountingKV ------------------------------------------------------------------------------


def test_counting_kv_counts_every_row_it_hands_out(fw):
    k, v = randn(6, 2, 4), randn(6, 2, 3, seed=1)
    store = CountingKV(fw, k, v)
    assert store.n == 6 and store.counters.snapshot() == {"rows_read": 0, "gather_calls": 0}
    idx = np.array([[0, 5, 2], [3, 3, 1]])
    k_rows, v_rows = store.gather(fw.tensor(idx))
    assert fw.is_tensor(k_rows) and fw.is_tensor(v_rows)
    np.testing.assert_array_equal(fw.numpy(k_rows), k[idx])
    np.testing.assert_array_equal(fw.numpy(v_rows), v[idx])
    assert fw.numpy(k_rows).shape == (2, 3, 2, 4) and fw.numpy(v_rows).dtype == np.float32
    # A repeated row is read twice.
    assert store.counters.snapshot() == {"rows_read": 6, "gather_calls": 1}
    # Reading every key and masking afterward is counted as reading every key.
    store.gather(list(range(6)))
    assert store.counters.rows_read == 12 and store.log == [((2, 3), 6), ((6,), 6)]
    # -1 is an index like any other: the last row, and it is counted.
    last_k, _ = store.gather(np.array([-1]))
    np.testing.assert_array_equal(fw.numpy(last_k)[0], k[5])
    assert store.counters.rows_read == 13
    # Neither the reference accessor nor the size is a read.
    ref_k, ref_v = store.reference()
    np.testing.assert_array_equal(ref_k, k)
    np.testing.assert_array_equal(ref_v, v)
    ref_k[:] = 0.0  # a copy
    np.testing.assert_array_equal(store.reference()[0], k)
    assert store.counters.rows_read == 13 and "n=6" in repr(store)


def test_counting_kv_valid_slots_are_the_only_ones_read(fw):
    k, v = randn(5, 3), randn(5, 3, seed=1)
    store = CountingKV(fw, k, v)
    idx = np.array([[4, 1, -1, -1], [0, -1, -1, -1], [2, 3, 4, 0]])
    k_rows, v_rows = store.gather(fw.tensor(idx), valid=fw.tensor(idx >= 0))
    # Cost: the number of valid slots, exactly.
    assert store.counters.rows_read == 7 and store.log == [((3, 4), 7)]
    got = fw.numpy(k_rows)
    np.testing.assert_array_equal(got[idx >= 0], k[idx[idx >= 0]])
    assert not got[idx < 0].any() and not fw.numpy(v_rows)[idx < 0].any()  # zeros, not the last row
    # Without the mask the same call reads 12 rows, 5 of them the last one.
    k_wrong, _ = store.gather(idx)
    assert store.counters.rows_read == 7 + 12
    np.testing.assert_array_equal(fw.numpy(k_wrong)[0, 2], k[4])
    # A masked slot may hold anything, even an index that is out of range.
    store.gather(np.array([99, 1]), valid=np.array([False, True]))
    assert store.counters.rows_read == 20
    store.gather(np.zeros((0,), dtype=np.int64))  # nothing to read is allowed
    assert store.counters.snapshot() == {"rows_read": 20, "gather_calls": 4}


def test_counting_kv_append_branches_and_refuses_misuse(fw):
    k, v = randn(4, 2, 3), randn(4, 2, 3, seed=1)
    store = CountingKV(fw, k[:2], v[:2])
    longer = store.append(fw.tensor(k[2:]), fw.tensor(v[2:]))
    assert (store.n, longer.n) == (2, 4)  # the original is untouched
    np.testing.assert_array_equal(longer.reference()[0], k)
    # One set of counters for the whole lineage: a decode loop is measured as a whole.
    longer.gather([3])
    store.gather([0, 1])
    assert store.counters is longer.counters and store.counters.rows_read == 3
    assert store.log is longer.log and len(store.log) == 2
    # A store that starts empty takes its row shapes from the first append.
    empty = CountingKV.empty(fw)
    assert empty.n == 0
    grown = empty.append(k[:1], v[:1]).append(k[1:], v[1:])
    np.testing.assert_array_equal(grown.reference()[1], v)
    assert grown.counters is empty.counters
    # A twin for the oracle: same rows, its own counters.
    twin = longer.with_framework(fw)
    twin.gather([0, 1, 2, 3])
    assert twin.counters.rows_read == 4 and longer.counters.rows_read == 3 and twin.n == 4

    for bad in ([4], [-5]):
        with pytest.raises(ValueError, match="row indices must be in"):
            longer.gather(bad)
    with pytest.raises(ValueError, match="integers"):
        longer.gather(np.array([0.0]))
    with pytest.raises(ValueError, match="valid must be a boolean array"):
        longer.gather([0, 1], valid=np.array([True]))
    with pytest.raises(ValueError, match="rows must have shapes"):
        longer.append(randn(1, 2, 4), randn(1, 2, 3))
    with pytest.raises(ValueError, match="same number of rows"):
        CountingKV(fw, k, v[:3])
    assert longer.counters.rows_read == 3  # refused calls count nothing


# -- TinyDSA ---------------------------------------------------------------------------------


def index_scores(q_idx, k_idx, w):
    """The score of the `indexer-topk-select` spec, in float64 numpy."""
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    per_head = np.maximum(np.einsum("thd,sd->ths", q_idx, k_idx).astype(np.float64) * dim**-0.5, 0)
    return np.einsum("th,ths->ts", w.astype(np.float64) * heads**-0.5, per_head)


def select(scores, visible, k):
    """Each row's min(k, S) highest visible scores, best first, ties to the lowest index."""
    masked = np.where(visible, scores, NEG_INF)
    order = np.argsort(-masked, axis=-1, kind="stable")[:, : min(k, scores.shape[1])]
    return np.where(np.take_along_axis(masked, order, axis=-1) == NEG_INF, -1, order)


def clear_margin(dsa, layer, x, k, margin=1e-4):
    """Whether no row's k-th and (k+1)-th scores are closer than `margin` unless equal."""
    scores = -np.sort(-dsa.index_scores_ref(layer, x).astype(np.float64), axis=-1)
    if k >= scores.shape[1]:
        return True
    both = np.isfinite(scores[:, k - 1]) & np.isfinite(scores[:, k])
    gap = scores[both, k - 1] - scores[both, k]
    return bool(((gap == 0) | (gap > margin)).all())


def test_tiny_dsa_pieces_have_the_documented_shapes_and_counts(fw):
    dsa = TinyDSA(fw, seed=1)
    assert dsa.types == ("full", "shared", "full", "shared") == index_types(4)
    assert index_types(8, freq=4, offset=3) == (
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
    )  # fmt: skip  (GLM-5.3)
    x = randn(7, 32)
    k_idx = dsa.index_key(1, fw.tensor(x))
    q_idx, w = dsa.index_query(1, fw.tensor(x))
    q, k, v = dsa.qkv(1, fw.tensor(x))
    assert all(fw.is_tensor(t) for t in (k_idx, q_idx, w, q, k, v))
    assert fw.numpy(k_idx).shape == (7, 8) and fw.numpy(q_idx).shape == (7, 4, 8)
    assert fw.numpy(w).shape == (7, 4)
    assert fw.numpy(q).shape == fw.numpy(k).shape == fw.numpy(v).shape == (7, 2, 8)
    out = dsa.out(1, v)
    assert fw.numpy(out).shape == (7, 32) and fw.numpy(out).dtype == np.float32
    # Only index keys are counted among the pieces.
    assert dsa.counters.snapshot() == {"indexer_calls": 0, "index_key_rows": 7}
    idx = dsa.indexer(1, fw.tensor(x))
    assert fw.numpy(idx).shape == (7, 4) and fw.numpy(idx).dtype.kind == "i"
    # An indexer call computes its own keys: no way around the count.
    assert dsa.counters.snapshot() == {"indexer_calls": 1, "index_key_rows": 14}
    assert fw.numpy(dsa.attend(1, fw.tensor(x), idx)).shape == (7, 32)
    assert dsa.counters.snapshot() == {"indexer_calls": 1, "index_key_rows": 14}
    # Rows are independent in every piece: a token's key does not depend on its batch.
    np.testing.assert_allclose(
        fw.numpy(dsa.index_key(1, fw.tensor(x[3:5]))), fw.numpy(k_idx)[3:5], rtol=1e-5, atol=1e-6
    )
    # Layers differ, seeds differ, a twin is the same model with its own counters.
    assert not np.allclose(fw.numpy(dsa.index_key(0, x)), fw.numpy(k_idx))
    assert not np.allclose(fw.numpy(TinyDSA(fw, seed=2).qkv(1, x)[0]), fw.numpy(q))
    twin = dsa.with_framework(fw)
    np.testing.assert_array_equal(fw.numpy(twin.qkv(1, x)[0]), fw.numpy(q))
    assert twin.counters.snapshot() == {"indexer_calls": 0, "index_key_rows": 0}


def test_tiny_dsa_indexer_is_the_spec_formula_on_its_own_pieces(fw):
    dsa = TinyDSA(fw, seed=3, top_k=5)
    x = randn(12, 32, seed=2)
    for layer in range(4):
        k_idx = fw.numpy(dsa.index_key(layer, x))
        q_idx, w = (fw.numpy(t) for t in dsa.index_query(layer, x))
        # The key is a LayerNorm (mean removed) with a weight and a bias.
        scores = index_scores(q_idx, k_idx, w)
        visible = np.tril(np.ones((12, 12), dtype=bool))
        ref = dsa.index_scores_ref(layer, x)
        np.testing.assert_allclose(ref[visible], scores[visible], rtol=1e-4, atol=1e-5)
        assert (ref[~visible] == NEG_INF).all()
        for k in (1, 3, 5, 12, 40):
            if not clear_margin(dsa, layer, x, k):
                continue
            got = fw.numpy(dsa.indexer(layer, x, k))
            np.testing.assert_array_equal(got, select(ref, visible, k))
            np.testing.assert_array_equal(got, dsa.indexer_ref(layer, x, k))
            assert got.shape == (12, min(k, 12))
            # Causal, no repeats, -1 only where a row has fewer than k visible tokens.
            for t, row in enumerate(got):
                real = row[row >= 0]
                assert len(real) == min(k, t + 1) == len(set(real)) and (real <= t).all()
                assert (row[len(real) :] == -1).all()
    # The ReLU makes exact zeros, hence exact ties: the kit settles them by index.
    assert (dsa.index_scores_ref(0, x) == 0).sum() > 0


def test_tiny_dsa_selection_that_covers_everything_is_full_causal_attention(fw):
    dsa = TinyDSA(fw, seed=4)
    x = randn(9, 32, seed=3)
    for layer in range(2):
        q, k, v = (fw.numpy(t) for t in dsa.qkv(layer, x))
        # (T, H, d) -> (H, T, d) for ops.attention, with a causal mask.
        causal = np.tril(np.ones((9, 9), dtype=bool))
        full = ops.attention(*(t.transpose(1, 0, 2) for t in (q, k, v)), causal).transpose(1, 0, 2)
        want = fw.numpy(dsa.out(layer, full))
        everything = dsa.indexer(layer, x, 9)  # k >= T: every visible token is selected
        np.testing.assert_allclose(
            fw.numpy(dsa.attend(layer, x, everything)), want, rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(dsa.layer_ref(layer, x, 99), want, rtol=1e-4, atol=1e-5)
        # With k < T the layer is a different function: the selection matters.
        assert np.abs(dsa.layer_ref(layer, x, 2) - want).max() > 1e-2
        # And whose selection is used matters (the "nearest full layer" mutants).
        other = dsa.indexer_ref(1 - layer, x, 3)
        assert np.abs(dsa.attend_ref(layer, x, other) - dsa.layer_ref(layer, x, 3)).max() > 1e-3


def dsa_step(fw, x_new, dsa, layer, cache, k):
    """A decode step as the `indexer-cache-decode` spec asks for it, on the kit's pieces.

    `cache` is `(index_keys, store)`: numpy indexer keys of the tokens so far and a
    CountingKV of their main keys and values.
    """
    index_keys, store = cache
    seen, new = store.n, x_new.shape[0]
    index_keys = np.concatenate((index_keys, fw.numpy(dsa.index_key(layer, x_new))))
    q, k_new, v_new = dsa.qkv(layer, x_new)
    store = store.append(k_new, v_new)
    q_idx, w = (fw.numpy(t) for t in dsa.index_query(layer, x_new))
    visible = np.arange(seen + new)[None, :] <= np.arange(seen, seen + new)[:, None]
    idx = select(index_scores(q_idx, index_keys, w).astype(np.float32), visible, k)
    k_rows, v_rows = store.gather(fw.tensor(idx), valid=fw.tensor(idx >= 0))
    q, k_rows, v_rows = fw.numpy(q), fw.numpy(k_rows), fw.numpy(v_rows)
    scores = np.einsum("thd,tjhd->thj", q, k_rows) / math.sqrt(q.shape[-1])
    scores = np.where((idx >= 0)[:, None, :], scores, NEG_INF)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    attended = np.einsum("thj,tjhd->thd", weights, v_rows).astype(np.float32)
    return fw.numpy(dsa.out(layer, fw.tensor(attended))), (index_keys, store)


def test_tiny_dsa_decoding_with_an_indexer_cache_equals_the_full_pass(fw):
    dsa = TinyDSA(fw, seed=5)
    layer, k, steps = 2, 3, 14
    seed = 0
    while True:  # a fixture whose selections do not hinge on float rounding
        x = randn(steps, 32, seed=seed)
        if clear_margin(dsa, layer, x, k):
            break
        seed += 1
    want = dsa.layer_ref(layer, x, k)
    for cuts in ([14], [5, 6, 7, 14], list(range(1, 15))):
        dsa.counters.reset()
        store = CountingKV.empty(fw)
        cache, pieces, start = (np.zeros((0, dsa.index_dim), np.float32), store), [], 0
        for stop in cuts:
            piece, cache = dsa_step(fw, x[start:stop], dsa, layer, cache, k)
            pieces.append(piece)
            start = stop
        np.testing.assert_allclose(np.concatenate(pieces), want, rtol=1e-4, atol=1e-5)
        # Two caches that grow together, at the cost the spec states.
        assert cache[1].n == steps and cache[0].shape == (steps, dsa.index_dim)
        assert dsa.counters.snapshot() == {"indexer_calls": 0, "index_key_rows": steps}
        valid_slots = sum(min(k, t + 1) for t in range(steps))
        assert store.counters.rows_read == valid_slots <= k * steps
        assert all(read <= k * shape[0] for shape, read in store.log)
    # Recomputing the indexer keys of the whole prefix each step is visible in the count.
    dsa.counters.reset()
    for stop in range(1, steps + 1):
        dsa.index_key(layer, x[:stop])
    assert dsa.counters.index_key_rows == steps * (steps + 1) // 2


def test_tiny_dsa_stack_shares_indices_with_the_nearest_full_layer(fw):
    dsa = TinyDSA(fw, seed=6, n_layers=6, top_k=3, types=["full", "shared", "shared", "full", "shared", "full"])  # fmt: skip
    h0 = randn(10, 32, seed=1)
    want = dsa.forward_ref(h0)

    def forward(types, nearest=True, k=None):
        h, idx, first = fw.tensor(h0), None, None
        for i, kind in enumerate(types):
            if kind == "full":
                idx = dsa.indexer(i, h, k)
                first = idx if first is None else first
            h = h + dsa.attend(i, h, idx if nearest else first)
        return fw.numpy(h)

    np.testing.assert_allclose(forward(dsa.types), want, rtol=1e-4, atol=1e-4)
    # Cost: one indexer call per full layer.
    assert dsa.counters.indexer_calls == 3 and dsa.counters.index_key_rows == 30
    # Reusing the first full layer's indices everywhere is a different function...
    assert np.abs(forward(dsa.types, nearest=False) - want).max() > 1e-3
    # ...and so is running every indexer, which also shows in the count.
    dsa.counters.reset()
    assert np.abs(forward(["full"] * 6) - want).max() > 1e-3
    assert dsa.counters.indexer_calls == 6
    np.testing.assert_allclose(dsa.forward_ref(h0, ["full"] * 6), forward(["full"] * 6), rtol=1e-4, atol=1e-4)  # fmt: skip
    np.testing.assert_allclose(dsa.forward_ref(h0, k=10), forward(dsa.types, k=10), rtol=1e-4, atol=1e-4)  # fmt: skip
    with pytest.raises(ValueError, match="no full layer comes before"):
        dsa.forward_ref(h0, ["shared"] + ["full"] * 5)
    with pytest.raises(ValueError, match="per layer"):
        dsa.forward_ref(h0, ["full"])


def test_tiny_dsa_refuses_misuse(fw):
    dsa = TinyDSA(fw, seed=0)
    x = randn(5, 32)
    with pytest.raises(ValueError, match="outside"):
        dsa.index_key(4, x)
    with pytest.raises(TypeError, match="must be an int"):
        dsa.qkv("0", x)
    with pytest.raises(ValueError, match="float rows of shape"):
        dsa.index_key(0, randn(5, 16))
    with pytest.raises(ValueError, match="positive int"):
        dsa.indexer(0, x, 0)
    with pytest.raises(ValueError, match="idx must be integers"):
        dsa.attend(0, x, np.zeros((4, 2), dtype=np.int64))
    with pytest.raises(ValueError, match="-1 or in"):
        dsa.attend(0, x, np.full((5, 2), 5))
    with pytest.raises(ValueError, match="attended must have shape"):
        dsa.out(0, randn(5, 16))
    with pytest.raises(ValueError, match="per layer"):
        TinyDSA(fw, types=["full"])
    assert dsa.counters.snapshot() == {"indexer_calls": 0, "index_key_rows": 0}
    # A row whose slots are all unused attends to nothing and gets the zero vector.
    assert not fw.numpy(dsa.attend(0, x, np.full((5, 2), -1))).any()


# -- the MTP chain ----------------------------------------------------------------------------


def test_mtp_params_extend_decoder_params():
    cfg = DecoderConfig(n_layers=2)
    params, main = mtp_params(cfg, n_layers=3, seed=4), decoder_params(cfg, seed=4)
    assert all(np.array_equal(params[name], main[name]) for name in main)
    extra = [name for name in params if name not in main]
    assert all(name.startswith("mtp.layers.") for name in extra) and len(extra) == 3 * 13
    stage = mtp_layer_params(params, 1)
    assert {name: value.shape for name, value in stage.items()} == {
        "enorm.weight": (32,), "hnorm.weight": (32,), "eh_proj.weight": (32, 64),
        "mtp_block.input_layernorm.weight": (32,),
        "mtp_block.self_attn.q_proj.weight": (32, 32),
        "mtp_block.self_attn.k_proj.weight": (16, 32),
        "mtp_block.self_attn.v_proj.weight": (16, 32),
        "mtp_block.self_attn.o_proj.weight": (32, 32),
        "mtp_block.post_attention_layernorm.weight": (32,),
        "mtp_block.mlp.gate_proj.weight": (48, 32),
        "mtp_block.mlp.up_proj.weight": (48, 32),
        "mtp_block.mlp.down_proj.weight": (32, 48),
        "post_norm.weight": (32,),
    }  # fmt: skip
    assert not np.array_equal(
        stage["eh_proj.weight"], mtp_layer_params(params, 2)["eh_proj.weight"]
    )
    with pytest.raises(KeyError):
        mtp_layer_params(params, 3)
    with pytest.raises(ValueError, match="at least 1"):
        mtp_params(cfg, 0)


def test_decoder_hidden_ref_is_what_the_head_reads():
    cfg = DecoderConfig(n_layers=2)
    params = decoder_params(cfg, seed=1)
    tokens = np.random.default_rng(0).integers(0, 64, (2, 7))
    hidden = decoder_hidden_ref(cfg, params, tokens)
    assert hidden.shape == (2, 7, 32) and hidden.dtype == np.float32
    np.testing.assert_allclose(
        hidden @ params["model.embed_tokens.weight"].T,
        decoder_forward_ref(cfg, params, tokens), rtol=1e-5, atol=1e-5,
    )  # fmt: skip
    np.testing.assert_allclose(decoder_hidden_ref(cfg, params, tokens[0]), hidden[0], rtol=1e-5, atol=1e-6)  # fmt: skip


def mtp_setup(seed=2, n_layers=3, length=9):
    cfg = DecoderConfig(n_layers=2)
    params = mtp_params(cfg, n_layers, seed)
    tokens = np.random.default_rng(seed).integers(0, cfg.vocab_size, length)
    hidden = decoder_hidden_ref(cfg, params, tokens)
    next_token = int(decoder_forward_ref(cfg, params, tokens)[-1].argmax())
    return cfg, params, tokens, next_token, hidden


def test_mtp_draft_is_a_chain_of_one_block_stages():
    cfg, params, tokens, next_token, hidden = mtp_setup()
    draft, logits = mtp_draft_ref(cfg, params, tokens, next_token, hidden, 3)
    assert draft.shape == (3,) and draft.dtype == np.int64 and logits.shape == (3, 64)
    assert logits.dtype == np.float32 and 1.0 < logits.std() < 4.0
    np.testing.assert_array_equal(draft, logits.argmax(-1))
    # Fewer stages are a prefix of more stages.
    for n in (1, 2):
        shorter, shorter_logits = mtp_draft_ref(cfg, params, tokens, next_token, hidden, n)
        np.testing.assert_array_equal(shorter, draft[:n])
        np.testing.assert_array_equal(shorter_logits, logits[:n])

    # Stage 0 by hand, from the module docstring.
    layer = mtp_layer_params(params, 0)
    embedding = params["model.embed_tokens.weight"]
    ids = np.concatenate((tokens[1:], [next_token]))
    joined = np.concatenate(
        (ops.rms_norm(embedding[ids], layer["enorm.weight"]), ops.rms_norm(hidden, layer["hnorm.weight"])),
        axis=-1,
    )  # fmt: skip
    x = joined @ layer["eh_proj.weight"].T
    block = {n[len("mtp_block.") :]: v for n, v in layer.items() if n.startswith("mtp_block.")}
    seq = len(tokens)
    h = ops.rms_norm(x, block["input_layernorm.weight"])
    q = (h @ block["self_attn.q_proj.weight"].T).reshape(seq, 4, 8).transpose(1, 0, 2)
    k = (h @ block["self_attn.k_proj.weight"].T).reshape(seq, 2, 8).transpose(1, 0, 2)
    v = (h @ block["self_attn.v_proj.weight"].T).reshape(seq, 2, 8).transpose(1, 0, 2)
    positions = np.arange(1, seq + 1)  # one ahead of the main model
    attended = ops.attention(
        ops.rope(q[None], positions), ops.rope(k[None], positions), v[None],
        np.tril(np.ones((seq, seq), dtype=bool)),
    )[0]  # fmt: skip
    x = x + attended.transpose(1, 0, 2).reshape(seq, 32) @ block["self_attn.o_proj.weight"].T
    x = x + ops.swiglu(
        ops.rms_norm(x, block["post_attention_layernorm.weight"]),
        block["mlp.gate_proj.weight"], block["mlp.up_proj.weight"], block["mlp.down_proj.weight"],
    )  # fmt: skip
    first = ops.rms_norm(x, layer["post_norm.weight"])[-1] @ embedding.T
    np.testing.assert_allclose(logits[0], first, rtol=1e-4, atol=1e-4)


def test_mtp_draft_is_sensitive_to_what_the_spec_mutants_break():
    cfg, params, tokens, next_token, hidden = mtp_setup(seed=5)
    _, logits = mtp_draft_ref(cfg, params, tokens, next_token, hidden, 3)

    def changed(**overrides):
        args = {"tokens": tokens, "next_token": next_token, "hidden": hidden, "params": params}
        args.update(overrides)
        return mtp_draft_ref(cfg, args["params"], args["tokens"], args["next_token"], args["hidden"], 3)[1]  # fmt: skip

    # The token the main model just produced is an input of stage 0.
    assert np.abs(changed(next_token=(next_token + 1) % 64) - logits)[0].max() > 1e-2
    # The hidden states are too, and stage 0 reads tokens[1:], not tokens[0].
    assert np.abs(changed(hidden=hidden[::-1].copy()) - logits)[0].max() > 1e-2
    other_first = tokens.copy()
    other_first[0] = (tokens[0] + 1) % 64
    np.testing.assert_array_equal(changed(tokens=other_first), logits)
    # Concatenation order: swapping the halves of eh_proj is another module.
    for stage in range(3):
        name = f"mtp.layers.{stage}.eh_proj.weight"
        swapped = dict(params) | {
            name: np.concatenate((params[name][:, 32:], params[name][:, :32]), 1)
        }
        assert np.abs(changed(params=swapped) - logits)[stage].max() > 1e-2
        for norm in ("enorm", "hnorm", "post_norm"):
            name = f"mtp.layers.{stage}.{norm}.weight"
            plain = dict(params) | {name: np.ones_like(params[name])}
            assert np.abs(changed(params=plain) - logits)[stage].max() > 1e-3, (stage, norm)
    # Stage 1 reads stage 0's output: its logits change when only stage 0's block changes.
    name = "mtp.layers.0.mtp_block.mlp.down_proj.weight"
    assert np.abs(changed(params=dict(params) | {name: -params[name]}) - logits)[1].max() > 1e-2
    with pytest.raises(ValueError, match="hidden must have shape"):
        mtp_draft_ref(cfg, params, tokens, next_token, hidden[:-1], 1)
    with pytest.raises(ValueError, match="T >= 1"):
        mtp_draft_ref(cfg, params, np.zeros((0,), np.int64), next_token, hidden[:0], 1)
    with pytest.raises(KeyError):
        mtp_draft_ref(cfg, params, tokens, next_token, hidden, 4)  # only 3 stages exist


# -- BlockDenoiser ----------------------------------------------------------------------------


def test_block_denoiser_counts_and_shapes(fw):
    model = BlockDenoiser(fw, seed=1)
    assert (model.vocab_size, model.pad_id, model.eos_id) == (32, 0, 1)
    cache = model.encode(fw.tensor(np.array([3, 7, 9, 4])))
    assert isinstance(cache, DenoiserCache) and cache.length == 4
    logits = model.denoise(fw.tensor(np.array([5, 5, 6, 30, 2, 9])), cache)
    assert fw.is_tensor(logits)
    assert fw.numpy(logits).shape == (6, 32) and fw.numpy(logits).dtype == np.float32
    assert model.counters.snapshot() == {
        "encoder_calls": 1, "encoder_tokens": 4, "denoise_calls": 1, "denoise_tokens": 6,
    }  # fmt: skip
    longer = model.encode([8, 8], cache)
    assert (cache.length, longer.length) == (4, 6)  # functional: the old cache still works
    np.testing.assert_array_equal(
        fw.numpy(model.denoise([5, 5, 6, 30, 2, 9], cache)), fw.numpy(logits)
    )
    assert model.counters.snapshot() == {
        "encoder_calls": 2, "encoder_tokens": 6, "denoise_calls": 2, "denoise_tokens": 12,
    }  # fmt: skip
    twin = model.with_framework(fw)
    np.testing.assert_array_equal(
        fw.numpy(twin.denoise([5, 5, 6, 30, 2, 9], cache)), fw.numpy(logits)
    )
    assert twin.counters.denoise_calls == 1 and model.counters.denoise_calls == 2
    for call, match in [
        (lambda: model.encode([]), "n >= 1"),
        (lambda: model.encode([32]), "ids must be in"),
        (lambda: model.encode([0.5]), "integers"),
        (lambda: model.denoise([1, 2], None), "DenoiserCache"),
        (lambda: model.denoise([1, 2], cache, np.zeros((3, 32), np.float32)), "self_cond_logits"),
        (lambda: BlockDenoiser(fw, max_context=5).encode([1] * 6), "max_context"),
    ]:
        with pytest.raises(ValueError, match=match):
            call()
    assert model.counters.denoise_calls == 2 and model.counters.encoder_calls == 2


def test_block_denoiser_cache_is_bit_identical_however_the_prefix_is_split(fw):
    model = BlockDenoiser(fw, seed=2)
    tokens = np.random.default_rng(0).integers(0, 32, 21)
    whole = model.encode(tokens)
    canvas = np.arange(8)
    want = fw.numpy(model.denoise(canvas, whole))
    for cuts in ([4, 12, 20, 21], [1, 21], list(range(1, 22))):
        cache, start = None, 0
        for stop in cuts:
            cache = model.encode(tokens[start:stop], cache)
            start = stop
        np.testing.assert_array_equal(cache.rows, whole.rows)
        np.testing.assert_array_equal(fw.numpy(model.denoise(canvas, cache)), want)
    # The encoder is causal: a row of the cache depends on the tokens up to it.
    np.testing.assert_array_equal(model.encode(tokens[:9]).rows, whole.rows[:9])
    # A different prefix, or the same canvas one block later, gives other logits.
    other = tokens.copy()
    other[3] = (other[3] + 1) % 32
    assert np.abs(fw.numpy(model.denoise(canvas, model.encode(other))) - want).max() > 1e-2
    assert np.abs(fw.numpy(model.denoise(canvas, model.encode(tokens[:-1]))) - want).max() > 1e-2


def test_block_denoiser_is_bidirectional_and_self_conditioned(fw):
    model = BlockDenoiser(fw, seed=3)
    cache = model.encode([3, 7, 9, 4, 11])
    canvas = np.array([5, 14, 6, 30, 2, 9, 21, 8])
    base = fw.numpy(model.denoise(canvas, cache))
    # Changing one canvas token changes the logits of earlier and of later positions.
    for position in (0, 4, 7):
        changed = canvas.copy()
        changed[position] = (changed[position] + 3) % 32
        delta = np.abs(fw.numpy(model.denoise(changed, cache)) - base).max(-1)
        assert (delta > 1e-4).all(), (position, delta)
    # A token on the canvas votes for itself.
    assert (base[np.arange(8), canvas] > fw.numpy(model.denoise((canvas + 1) % 32, cache))[np.arange(8), canvas]).all()  # fmt: skip
    # Self-conditioning: the previous step's favorite gains, position by position.
    previous = np.full((8, 32), -4.0, dtype=np.float32)
    previous[np.arange(8), [10, 11, 12, 13, 14, 15, 16, 17]] = 6.0
    conditioned = fw.numpy(model.denoise(canvas, cache, fw.tensor(previous)))
    gain = conditioned - base
    assert (gain[np.arange(8), [10, 11, 12, 13, 14, 15, 16, 17]] > 4.0).all()
    assert np.abs(gain).max() < 3.0 * 2.0 + 1e-3  # sharpness * SELF_COND_VOTE at most
    # The knobs: sharpness scales the votes, eos_bias moves one column.
    sharp = BlockDenoiser(fw, seed=3, sharpness=6.0)
    np.testing.assert_allclose(
        fw.numpy(sharp.denoise(canvas, cache)), 2.0 * base, rtol=1e-5, atol=1e-5
    )
    biased = fw.numpy(BlockDenoiser(fw, seed=3, eos_bias=-2.5).denoise(canvas, cache))
    np.testing.assert_allclose(biased[:, 1], base[:, 1] - 2.5, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.delete(biased, 1, axis=1), np.delete(base, 1, axis=1))
    assert not np.allclose(
        fw.numpy(BlockDenoiser(fw, seed=4).denoise(canvas, cache)), base, atol=1e-2
    )


def diffusion_generate(fw, model, prompt, rng, *, length=8, max_steps=12, max_canvases=3,
                       stability=2, confidence=0.5, t_min=0.2, t_max=1.0, bound=1.0):  # fmt: skip
    """The `block-diffusion-decode` loop of the spec, with numpy doing the sampling."""
    vocab = model.vocab_size
    cache = model.encode(prompt)
    out, steps_per_canvas = list(prompt), []
    for block in range(max_canvases):
        canvas, previous, history = rng.integers(0, vocab, length), None, []
        for remaining in range(max_steps, 0, -1):
            raw = fw.numpy(model.denoise(fw.tensor(canvas), cache, previous))
            scaled = raw / np.float32(t_min + (t_max - t_min) * remaining / max_steps)
            cdf = np.cumsum(ops.softmax(scaled), axis=-1)
            sampled = np.minimum((cdf < rng.random(length)[:, None]).sum(-1), vocab - 1)
            argmax = scaled.argmax(-1)
            accepted = ops.eb_accept(scaled, bound)
            canvas = np.where(accepted, sampled, rng.integers(0, vocab, length))
            logp = ops.log_softmax(scaled)
            entropy = float(-(np.exp(logp) * logp).sum(-1).mean())
            stable = len(history) >= stability and all(
                np.array_equal(past, argmax) for past in history[-stability:]
            )
            history.append(argmax)
            previous = fw.tensor(scaled)
            if stable and entropy < confidence:
                break
        steps_per_canvas.append(len(history))
        final = argmax.copy()
        if (final == model.eos_id).any():
            final[int(np.argmax(final == model.eos_id)) + 1 :] = model.pad_id
            out += final.tolist()
            break
        out += final.tolist()
        if block + 1 < max_canvases:  # the last canvas is never read again
            cache = model.encode(fw.tensor(final), cache)
    return out, steps_per_canvas


def test_block_denoiser_sharpness_decides_whether_a_canvas_converges(fw):
    prompt = [3, 7, 9, 4]
    for seed in (0, 1, 2):
        # Sharp: every canvas stops as early as the rule allows, or one step later.
        model = BlockDenoiser(fw, seed=seed, sharpness=8.0, eos_bias=-100.0)
        out, steps = diffusion_generate(fw, model, prompt, np.random.default_rng(seed))
        assert len(steps) == 3 and max(steps) <= 5 and min(steps) >= 3, steps
        assert len(out) == 4 + 3 * 8 and model.eos_id not in out
        # Cost of the loop, as the spec states it: the prompt once, then each finished
        # canvas once (the last one is never encoded).
        assert model.counters.encoder_tokens == 4 + (3 - 1) * 8
        assert model.counters.encoder_calls == 3
        assert model.counters.denoise_calls == sum(steps)
        # Flat: the entropy never gets under the threshold; every canvas runs out of steps.
        flat = BlockDenoiser(fw, seed=seed, sharpness=0.5, eos_bias=-100.0)
        _, steps = diffusion_generate(fw, flat, prompt, np.random.default_rng(seed))
        assert steps == [12, 12, 12]
        assert flat.counters.denoise_calls == 36
    # An end-of-sequence token ends the text inside a canvas and pads the rest.
    eager = BlockDenoiser(fw, seed=0, sharpness=3.0, eos_bias=100.0)
    out, steps = diffusion_generate(fw, eager, prompt, np.random.default_rng(0))
    assert len(steps) == 1 and out == [*prompt, eager.eos_id, *[eager.pad_id] * 7]
    assert eager.counters.encoder_tokens == 4
    # A milder bias ends some texts in a later canvas, at a position inside it.
    ends = []
    for seed in range(6):
        mild = BlockDenoiser(fw, seed=seed, sharpness=3.0, eos_bias=3.0)
        out, steps = diffusion_generate(fw, mild, prompt, np.random.default_rng(seed))
        if mild.eos_id in out[4:]:
            position = out[4:].index(mild.eos_id)
            assert all(token == mild.pad_id for token in out[4:][position + 1 :])
            assert len(out) == 4 + 8 * len(steps)
            ends.append(position)
    assert len(set(ends)) >= 2, ends


def test_block_denoiser_generation_is_the_same_tokens_in_every_framework():
    outs = []
    for name in ("torch", "jax", "mlx"):
        try:
            fw = frameworks.get(name)
        except frameworks.FrameworkUnavailable:
            continue
        model = BlockDenoiser(fw, seed=1, sharpness=2.0, eos_bias=-2.0)
        outs.append(diffusion_generate(fw, model, [5, 6], np.random.default_rng(7)))
    if len(outs) < 2:
        pytest.skip("needs two frameworks")
    assert all(out == outs[0] for out in outs[1:])
