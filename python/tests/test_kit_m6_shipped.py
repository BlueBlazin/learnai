"""Milestone 6 kit against the shipped implementations it mirrors, by running them.

Each check builds the module of ``transformers`` (5.23.0) or ``mlx-lm`` (0.32.2) with a
tiny configuration, copies the kit's seeded weights into it, and compares outputs. No
weights are downloaded: every model here is a few thousand random numbers.

The functions named ``compare_*`` return the largest absolute difference they saw; the
tests assert on it. The differences measured on 2026-11-21 are recorded in
``docs/PROBLEM_AUTHORING.md`` (section 7, "Checked against the shipped code"); to print
them again::

    uv run --project python --extra jax --extra mlx --extra models python python/tests/test_kit_m6_shipped.py
"""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import pytest

from learnai.kit import (
    DecoderConfig,
    GDNConfig,
    HybridConfig,
    TinyDSA,
    decoder_forward_ref,
    decoder_hidden_ref,
    gdn_layer_ref,
    gdn_params,
    hybrid_forward_ref,
    hybrid_params,
    mtp_draft_ref,
    mtp_params,
    ops,
)

#: Norm weights that Qwen3.5 stores zero-centered (the model computes 1 + weight).
ZERO_CENTERED = (
    "input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight",
    "k_norm.weight", "model.norm.weight",
)  # fmt: skip


class NumpyFramework:
    """The smallest adapter: a kit object bound to it speaks numpy."""

    def is_tensor(self, value):
        return False

    def tensor(self, value, dtype=None):
        return np.asarray(value)

    def numpy(self, value):
        return np.asarray(value)


def torch_and_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    return torch


def qwen_text_config(cfg: HybridConfig, layer_types=None):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_mlp,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads, head_dim=cfg.head_dim, rms_norm_eps=cfg.norm_eps,
        max_position_embeddings=512, tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default", "rope_theta": cfg.rope_base,
            "partial_rotary_factor": cfg.partial_rotary_factor,
        },
        linear_conv_kernel_dim=cfg.conv_kernel, linear_key_head_dim=cfg.key_head_dim,
        linear_value_head_dim=cfg.value_head_dim, linear_num_key_heads=cfg.n_key_heads,
        linear_num_value_heads=cfg.n_value_heads,
        layer_types=list(cfg.layers if layer_types is None else layer_types),
    )  # fmt: skip


def tiny_qwen(cfg: HybridConfig, params: dict[str, np.ndarray]):
    """A ``Qwen3_5ForCausalLM`` holding the kit's weights (norm gains stored minus 1)."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM(qwen_text_config(cfg)).eval().float()
    expected = model.state_dict()
    state = {}
    for name, value in params.items():
        tensor = torch.from_numpy(value.copy())
        state[name] = tensor - 1.0 if name.endswith(ZERO_CENTERED) else tensor
        assert expected[name].shape == state[name].shape, name
    state["lm_head.weight"] = torch.from_numpy(params["model.embed_tokens.weight"].copy())
    assert set(expected) == set(state)  # the kit's names are the checkpoint's names
    model.load_state_dict(state)
    return model


# -- small pieces: l2norm, the convolution, partial RoPE ----------------------------------------


def test_l2norm_and_causal_conv_match_qwen3_5():
    torch = torch_and_transformers()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5, 12)).astype(np.float32)
    want = qwen.l2norm(torch.from_numpy(x)).numpy()
    assert np.abs(ops.l2norm(x) - want).max() < 1e-6

    # The prefill path (pad by K - 1, cut to T) and the one-token path (saved state).
    channels, kernel, steps = 10, 4, 9
    x = rng.standard_normal((2, steps, channels)).astype(np.float32)
    w = rng.standard_normal((channels, kernel)).astype(np.float32)
    with torch.no_grad():
        full = qwen.causal_conv1d_fn(torch.from_numpy(x).transpose(1, 2), torch.from_numpy(w))
    got, _state = ops.causal_conv1d(x, w)
    assert np.abs(got - full.transpose(1, 2).numpy()).max() < 1e-5
    conv_state = torch.zeros(2, channels, kernel - 1)
    mine = None
    for t in range(steps):
        with torch.no_grad():
            step = qwen.causal_conv1d_update(
                torch.from_numpy(x[:, t : t + 1]).transpose(1, 2), conv_state, torch.from_numpy(w)
            )
        piece, mine = ops.causal_conv1d(x[:, t : t + 1], w, mine)
        assert np.abs(piece - step.transpose(1, 2).numpy()).max() < 1e-5
        # Their state is (C, K - 1); the kit's is (K - 1, C). Same numbers.
        np.testing.assert_array_equal(mine, conv_state.transpose(1, 2).numpy())


def test_partial_rope_matches_qwen3_5_rotary_embedding():
    torch = torch_and_transformers()
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    for head_dim, factor, base in [(16, 0.25, 1e7), (32, 0.5, 1e4), (8, 1.0, 100.0)]:
        cfg = HybridConfig(head_dim=head_dim, partial_rotary_factor=factor, rope_base=base)
        rotary = qwen.Qwen3_5TextRotaryEmbedding(qwen_text_config(cfg))
        rng = np.random.default_rng(head_dim)
        q = rng.standard_normal((2, 3, 7, head_dim)).astype(np.float32)
        k = rng.standard_normal((2, 3, 7, head_dim)).astype(np.float32)
        positions = np.array([[0, 1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 200, 201, 202]])
        ids = torch.from_numpy(positions)[None].expand(3, -1, -1)  # text: the three grids agree
        cos, sin = rotary(torch.from_numpy(q), ids)
        want_q, want_k = qwen.apply_rotary_pos_emb(
            torch.from_numpy(q), torch.from_numpy(k), cos, sin
        )
        assert np.abs(ops.partial_rope(q, positions, base, factor) - want_q.numpy()).max() < 2e-5
        assert np.abs(ops.partial_rope(k, positions, base, factor) - want_k.numpy()).max() < 2e-5


# -- the gated delta layer against Qwen3_5GatedDeltaNet -----------------------------------------


def qwen_gdn(cfg: GDNConfig, params: dict[str, np.ndarray]):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    hybrid = HybridConfig(
        n_layers=1, layer_types=("linear_attention",), d_model=cfg.d_model,
        n_key_heads=cfg.n_key_heads, n_value_heads=cfg.n_value_heads,
        key_head_dim=cfg.key_head_dim, value_head_dim=cfg.value_head_dim,
        conv_kernel=cfg.conv_kernel, norm_eps=cfg.norm_eps,
    )  # fmt: skip
    config = qwen_text_config(hybrid)
    layer = Qwen3_5GatedDeltaNet(config, 0).eval().float()
    assert {n: tuple(v.shape) for n, v in layer.state_dict().items()} == {
        n: v.shape for n, v in params.items()
    }
    layer.load_state_dict({n: torch.from_numpy(v.copy()) for n, v in params.items()})
    return layer, config


def compare_gdn_chunked(cfg: GDNConfig, steps: int, seed: int = 0) -> float:
    """One multi-token call without a cache: transformers takes its chunked path."""
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    params = gdn_params(cfg, seed)
    layer, _ = qwen_gdn(cfg, params)
    x = np.random.default_rng(seed).standard_normal((2, steps, cfg.d_model)).astype(np.float32)
    with (
        mock.patch.object(qwen, "torch_chunk_gated_delta_rule", wraps=qwen.torch_chunk_gated_delta_rule) as chunked,
        mock.patch.object(qwen, "torch_recurrent_gated_delta_rule", wraps=qwen.torch_recurrent_gated_delta_rule) as step,
        torch.no_grad(),
    ):  # fmt: skip
        want = layer(torch.from_numpy(x)).numpy()
    assert chunked.call_count == 1 and step.call_count == 0
    got, _ = gdn_layer_ref(cfg, params, x)
    return float(np.abs(got - want).max())


def compare_gdn_cached(
    cfg: GDNConfig, prefill: int, single: int, seed: int = 0
) -> dict[str, float]:
    """A prefill call, then one token at a time with a cache: the recurrent path."""
    import torch
    from transformers import DynamicCache
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    params = gdn_params(cfg, seed)
    layer, config = qwen_gdn(cfg, params)
    steps = prefill + single
    x = np.random.default_rng(seed + 1).standard_normal((1, steps, cfg.d_model)).astype(np.float32)
    cache, state, worst = DynamicCache(config=config), None, 0.0
    cuts = [0, prefill, *range(prefill + 1, steps + 1)]
    with (
        mock.patch.object(qwen, "torch_chunk_gated_delta_rule", wraps=qwen.torch_chunk_gated_delta_rule) as chunked,
        mock.patch.object(qwen, "torch_recurrent_gated_delta_rule", wraps=qwen.torch_recurrent_gated_delta_rule) as step,
        torch.no_grad(),
    ):  # fmt: skip
        for start, stop in itertools.pairwise(cuts):
            want = layer(torch.from_numpy(x[:, start:stop]), cache_params=cache).numpy()
            got, state = gdn_layer_ref(cfg, params, x[:, start:stop], state)
            worst = max(worst, float(np.abs(got - want).max()))
    assert chunked.call_count == 1 and step.call_count == single
    # Their memory is (heads, d_key, d_value), the transpose of the kit's. Their conv
    # state is (channels, K): the last K projected inputs, one more than the next
    # token's convolution reads. The kit (like mlx-lm) keeps the K - 1 that matter.
    theirs = cache.layers[0]
    memory = theirs.recurrent_states[0].numpy().transpose(0, 1, 3, 2)
    assert theirs.conv_states[0].shape[-1] == cfg.conv_kernel
    conv = theirs.conv_states[0].numpy().transpose(0, 2, 1)[:, 1:]
    return {
        "output": worst,
        "memory": float(np.abs(memory - state[1]).max()),
        "conv_state": float(np.abs(conv - state[0]).max()),
    }


GDN_CASES = {
    "2k4v": GDNConfig(),
    "3k3v": GDNConfig(d_model=24, n_key_heads=3, n_value_heads=3, key_head_dim=6, value_head_dim=4),
    "1k4v_k2": GDNConfig(d_model=16, n_key_heads=1, n_value_heads=4, key_head_dim=8,
                         value_head_dim=6, conv_kernel=2),
}  # fmt: skip


@pytest.mark.parametrize("name", GDN_CASES)
def test_gated_delta_layer_matches_qwen3_5_chunked_path(name):
    torch_and_transformers()
    # 5 and 64 fit one chunk of 64 (5 is padded); 70 and 150 cross chunk boundaries.
    for steps in (1, 5, 64, 70, 150):
        assert compare_gdn_chunked(GDN_CASES[name], steps) < 2e-4, steps


@pytest.mark.parametrize("name", GDN_CASES)
def test_gated_delta_layer_matches_qwen3_5_recurrent_path_and_state(name):
    torch_and_transformers()
    diffs = compare_gdn_cached(GDN_CASES[name], prefill=70, single=12)
    assert diffs["output"] < 2e-4 and diffs["memory"] < 2e-4 and diffs["conv_state"] < 1e-5, diffs


@pytest.mark.parametrize("backend", ["torch", "jax", "mlx"])
def test_gated_delta_layer_in_each_framework_matches_qwen3_5(backend):
    torch = torch_and_transformers()
    from learnai.judge import frameworks

    try:
        fw = frameworks.get(backend)
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))
    cfg = GDNConfig()
    params = gdn_params(cfg, seed=3)
    layer, _ = qwen_gdn(cfg, params)
    x = np.random.default_rng(3).standard_normal((70, cfg.d_model)).astype(np.float32)
    with torch.no_grad():
        want = layer(torch.from_numpy(x)[None])[0].numpy()
    got, _ = ops.gated_delta_layer(fw.tensor(x), fw.to_framework(params), None, cfg.n_key_heads)
    assert np.abs(fw.numpy(got) - want).max() < 2e-4


# -- the hybrid forward against a tiny Qwen3.5 text decoder -------------------------------------


def compare_hybrid(cfg: HybridConfig, steps: int, seed: int = 0) -> dict[str, float]:
    """One pass, then prefill in chunks and token by token with a cache, on both sides."""
    import torch
    from transformers import DynamicCache

    params = hybrid_params(cfg, seed)
    model = tiny_qwen(cfg, params)
    tokens = np.random.default_rng(seed).integers(0, cfg.vocab_size, steps)
    with torch.no_grad():
        want = model(torch.from_numpy(tokens)[None]).logits[0].numpy()
    got, _ = hybrid_forward_ref(cfg, params, tokens)
    one_pass = float(np.abs(got - want).max())

    cuts = sorted({0, min(3, steps), steps // 2, max(steps - 3, 0), steps - 2, steps - 1, steps})
    cuts = [c for c in cuts if c >= 0]
    cache, state, theirs, mine = DynamicCache(config=model.config), None, [], []
    with torch.no_grad():
        for start, stop in itertools.pairwise(cuts):
            out = model(
                torch.from_numpy(tokens[start:stop])[None], past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            theirs.append(out.logits[0].numpy())
            piece, state = hybrid_forward_ref(cfg, params, tokens[start:stop], state)
            mine.append(piece)
    theirs, mine = np.concatenate(theirs), np.concatenate(mine)
    diffs = {
        "one_pass": one_pass,
        "cached": float(np.abs(mine - theirs).max()),
        "kit_cached_vs_one_pass": float(np.abs(mine - got).max()),
        "logit_scale": float(np.abs(want).max()),
        "argmax_agree": float((mine.argmax(-1) == theirs.argmax(-1)).mean()),
    }
    # The caches hold the same things: rotated keys and values, memories, conv states.
    for index, kind in enumerate(cfg.layers):
        layer = cache.layers[index]
        if kind == "full_attention":
            diffs["keys"] = max(
                diffs.get("keys", 0.0), float(np.abs(layer.keys[0].numpy() - state[index][0]).max())
            )
            diffs["values"] = max(
                diffs.get("values", 0.0),
                float(np.abs(layer.values[0].numpy() - state[index][1]).max()),
            )
        else:
            memory = layer.recurrent_states[0][0].numpy().transpose(0, 2, 1)
            diffs["memory"] = max(
                diffs.get("memory", 0.0), float(np.abs(memory - state[index][1]).max())
            )
    return diffs  # fmt: skip


HYBRID_CASES = {
    "default": HybridConfig(),  # 3 linear, 1 full
    "8_layers": HybridConfig(n_layers=8, partial_rotary_factor=0.5, rope_base=1e7),
    "mixed": HybridConfig(
        n_layers=5, layer_types=("full_attention", "linear_attention", "full_attention",
                                 "linear_attention", "linear_attention"),
        n_heads=2, n_kv_heads=1, head_dim=8, n_key_heads=2, n_value_heads=2, conv_kernel=3,
    ),
}  # fmt: skip


@pytest.mark.parametrize("name", HYBRID_CASES)
def test_hybrid_forward_matches_a_tiny_qwen3_5_decoder(name):
    torch_and_transformers()
    for steps in (6, 70, 150):
        diffs = compare_hybrid(HYBRID_CASES[name], steps)
        assert diffs["one_pass"] < 5e-4 and diffs["cached"] < 5e-4, (steps, diffs)
        assert max(diffs["keys"], diffs["values"], diffs["memory"]) < 5e-4, diffs
        assert diffs["logit_scale"] > 3.0  # the comparison is not between two zeros


def test_hybrid_schedule_matches_qwen3_5_config():
    torch_and_transformers()
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    for n_layers, interval in [(24, 4), (7, 3), (5, 1), (4, 6)]:
        theirs = Qwen3_5TextConfig(num_hidden_layers=n_layers, full_attention_interval=interval)
        ours = HybridConfig(n_layers=n_layers, full_attention_interval=interval)
        assert tuple(theirs.layer_types) == ours.layers


def compare_hybrid_mlx_lm(cfg: HybridConfig, steps: int, seed: int = 0) -> dict[str, float]:
    """The same model in ``mlx-lm`` (``qwen3_5.TextModel``), one pass and with its cache."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from mlx_lm.models import qwen3_5

    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5_text", hidden_size=cfg.d_model, intermediate_size=cfg.d_mlp,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads, head_dim=cfg.head_dim, rms_norm_eps=cfg.norm_eps,
        vocab_size=cfg.vocab_size, linear_num_value_heads=cfg.n_value_heads,
        linear_num_key_heads=cfg.n_key_heads, linear_key_head_dim=cfg.key_head_dim,
        linear_value_head_dim=cfg.value_head_dim, linear_conv_kernel_dim=cfg.conv_kernel,
        tie_word_embeddings=True, full_attention_interval=cfg.full_attention_interval,
        rope_parameters={
            "type": "default", "rope_theta": cfg.rope_base,
            "partial_rotary_factor": cfg.partial_rotary_factor,
        },
    )  # fmt: skip
    model = qwen3_5.TextModel(args)
    params = hybrid_params(cfg, seed)
    # mlx-lm keeps plain gains (its `sanitize` adds the 1) and a channels-last kernel.
    weights = [
        (name, mx.array(np.moveaxis(value, 2, 1) if name.endswith("conv1d.weight") else value))
        for name, value in params.items()
    ]
    model.update(tree_unflatten(weights))
    model.eval()
    tokens = np.random.default_rng(seed).integers(0, cfg.vocab_size, steps)
    want = np.array(model(mx.array(tokens)[None])[0])
    got, _ = hybrid_forward_ref(cfg, params, tokens)
    cache, pieces = model.make_cache(), []
    cuts = sorted({0, min(3, steps), steps // 2, steps - 2, steps - 1, steps})
    for start, stop in itertools.pairwise(cuts):
        pieces.append(np.array(model(mx.array(tokens[start:stop])[None], cache=cache)[0]))
    return {
        "one_pass": float(np.abs(got - want).max()),
        "cached": float(np.abs(got - np.concatenate(pieces)).max()),
        "logit_scale": float(np.abs(want).max()),
    }


def test_hybrid_forward_matches_mlx_lm_qwen3_5():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    for cfg in (HybridConfig(), HybridConfig(n_layers=8, partial_rotary_factor=0.5, rope_base=1e7)):
        for steps in (6, 70):
            diffs = compare_hybrid_mlx_lm(cfg, steps)
            assert diffs["one_pass"] < 5e-4 and diffs["cached"] < 5e-4, (steps, diffs)
            assert diffs["logit_scale"] > 3.0


# -- TinyDSA's indexer against GlmMoeDsaIndexer -------------------------------------------------

ROPE_DIM = 4  # the indexer's rotary slice; rotated by the identity below


def glm_indexer(dsa: TinyDSA, layer: int):
    import torch
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer

    config = GlmMoeDsaConfig(
        vocab_size=16, hidden_size=dsa.d_model, intermediate_size=8, moe_intermediate_size=8,
        num_hidden_layers=dsa.n_layers, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=2, num_experts_per_tok=1, kv_lora_rank=8,
        q_lora_rank=dsa.d_model,  # the kit projects indexer queries from x itself
        qk_rope_head_dim=ROPE_DIM, v_head_dim=8, qk_nope_head_dim=4, index_topk=dsa.top_k,
        index_head_dim=dsa.index_dim, index_n_heads=dsa.index_heads, max_position_embeddings=64,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )  # fmt: skip
    indexer = GlmMoeDsaIndexer(config, layer).eval()
    weights = dsa._layers[layer]
    indexer.load_state_dict(
        {
            "wq_b.weight": torch.from_numpy(weights["index_q"].copy()),
            "wk.weight": torch.from_numpy(weights["index_k"].copy()),
            "k_norm.weight": torch.from_numpy(weights["index_k_norm.weight"].copy()),
            "k_norm.bias": torch.from_numpy(weights["index_k_norm.bias"].copy()),
            "weights_proj.weight": torch.from_numpy(weights["index_w"].copy()),
        }
    )
    return indexer, config


def run_glm_indexer(indexer, x: np.ndarray, start: int, cache, k: int):
    """Indices, the scores they were the top-k of, and the causal mask, for rows of ``x``
    that continue ``start`` cached tokens."""
    import torch

    new, total = x.shape[0], start + x.shape[0]
    # TinyDSA has no positions: cos 1 and sin 0 make the indexer's rotation the identity
    # (up to its reordering of the rotated channels, the same for queries and keys).
    cos, sin = torch.ones(1, new, ROPE_DIM), torch.zeros(1, new, ROPE_DIM)
    mask = (torch.arange(total)[None, :] <= (start + torch.arange(new))[:, None])[None]
    seen = {}
    real_topk = torch.Tensor.topk

    def spy(self, *args, **kwargs):
        seen["scores"] = self.detach().clone()
        return real_topk(self, *args, **kwargs)

    indexer.index_topk = k
    rows = torch.from_numpy(x)[None]
    with mock.patch.object(torch.Tensor, "topk", spy), torch.no_grad():
        idx = indexer(rows, rows, (cos, sin), mask, None, past_key_values=cache)
    return idx[0].numpy(), seen["scores"][0].numpy(), mask[0].numpy()


def compare_indexer(seed: int, steps: int, layer: int) -> dict[str, float]:
    from transformers import DynamicCache

    dsa = TinyDSA(NumpyFramework(), seed=seed)
    indexer, config = glm_indexer(dsa, layer)
    x = np.random.default_rng(seed).standard_normal((steps, dsa.d_model)).astype(np.float32)
    mine = dsa.index_scores_ref(layer, x)
    finite = np.isfinite(mine)
    diffs = {"scores": 0.0, "rows_compared": 0.0, "rows_with_boundary_ties": 0.0}
    for k in (1, 4, 7, steps, steps + 10):
        idx, scores, mask = run_glm_indexer(indexer, x, 0, None, k)
        assert np.array_equal(np.isfinite(scores), finite)  # the same causal masking
        diffs["scores"] = max(diffs["scores"], float(np.abs(scores[finite] - mine[finite]).max()))
        assert idx.shape == (steps, min(k, steps))  # topk(min(k, S))
        chosen = dsa.indexer_ref(layer, x, k)
        for t in range(steps):
            theirs = {int(i) for i in idx[t] if mask[t, i]}  # intersected with the causal mask
            ours = {int(i) for i in chosen[t] if i >= 0}
            assert len(theirs) == len(ours) == min(k, t + 1)
            # The selected scores agree as multisets whatever the tie order...
            np.testing.assert_allclose(
                sorted(mine[t, list(theirs)]), sorted(mine[t, list(ours)]), rtol=1e-5, atol=1e-6
            )
            ranked = np.sort(mine[t, : t + 1])[::-1]
            tied = len(ours) < t + 1 and ranked[len(ours) - 1] - ranked[len(ours)] < 1e-6
            diffs["rows_with_boundary_ties"] += tied
            if not tied:  # ...and the selected tokens are the same when no tie is cut
                assert theirs == ours, (k, t)
                diffs["rows_compared"] += 1

    # Decoding with the indexer cache of the shipped model: a prefill chunk, single
    # tokens, another chunk. The kit's full causal pass must select the same tokens.
    cache, got = DynamicCache(config=config), []
    cuts = [0, 6, 7, 8, steps // 2 + 4, steps]
    for start, stop in itertools.pairwise(cuts):
        idx, _, mask = run_glm_indexer(indexer, x[start:stop], start, cache, 4)
        got += [{int(i) for i in idx[t] if mask[t, i]} for t in range(stop - start)]
    chosen = dsa.indexer_ref(layer, x, 4)
    for t in range(steps):
        ranked = np.sort(mine[t, : t + 1])[::-1]
        if t + 1 <= 4 or ranked[3] - ranked[4] >= 1e-6:
            assert got[t] == {int(i) for i in chosen[t] if i >= 0}, t
    # The cached indexer keys are the kit's keys with the rotary slice reordered
    # (evens, then odds), which the dot product does not see.
    keys = cache.layers[layer].indexer_keys[0].numpy()
    order = [*range(0, ROPE_DIM, 2), *range(1, ROPE_DIM, 2), *range(ROPE_DIM, dsa.index_dim)]
    diffs["cached_keys"] = float(np.abs(keys - dsa._index_key(layer, x)[:, order]).max())
    return diffs


def test_tiny_dsa_indexer_matches_glm_moe_dsa_indexer():
    torch_and_transformers()
    compared = 0.0
    for seed, layer in [(1, 0), (2, 2), (3, 3)]:
        diffs = compare_indexer(seed, steps=24, layer=layer)
        assert diffs["scores"] < 1e-5 and diffs["cached_keys"] < 1e-5, diffs
        compared += diffs["rows_compared"]
    assert compared > 300  # most rows have no tie at the cut and were compared exactly


def test_index_types_match_glm_moe_dsa_config():
    torch_and_transformers()
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    from learnai.kit import index_types

    for n_layers, freq, offset in [(8, 4, 3), (6, 2, 1), (5, 1, 2), (7, 3, 0), (4, 0, 5)]:
        theirs = GlmMoeDsaConfig(
            num_hidden_layers=n_layers, index_topk_freq=freq, index_skip_topk_offset=offset
        )
        assert tuple(theirs.indexer_types) == index_types(n_layers, freq, offset)


# -- the MTP chain against MtpLayer / MtpModel --------------------------------------------------


def tiny_llama_with_mtp(cfg: DecoderConfig, params: dict[str, np.ndarray], n_stages: int):
    """A Llama decoder and an ``MtpModel`` on top of it, holding the kit's weights.

    ``decoder_block_ref`` is a Llama block except for the RoPE layout: the kit pairs
    adjacent channels ``(2i, 2i + 1)``, Llama pairs ``(i, i + head_dim / 2)``. Reordering
    the rows of ``q_proj`` and ``k_proj`` inside each head (evens, then odds) maps one
    onto the other without changing any score.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.modeling_layers import MtpModel

    head_dim = cfg.head_dim
    within = np.concatenate((np.arange(0, head_dim, 2), np.arange(1, head_dim, 2)))

    def convert(name: str, value: np.ndarray) -> torch.Tensor:
        if name.endswith(("self_attn.q_proj.weight", "self_attn.k_proj.weight")):
            heads = value.shape[0] // head_dim
            value = value[np.concatenate([h * head_dim + within for h in range(heads)])]
        return torch.from_numpy(value.copy())

    config = LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model, intermediate_size=cfg.d_mlp,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads, rms_norm_eps=cfg.norm_eps,
        max_position_embeddings=128, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_base},
    )  # fmt: skip
    config.num_mtp_layers = n_stages
    main = LlamaForCausalLM(config).eval()
    state = {n: convert(n, v) for n, v in params.items() if not n.startswith("mtp.")}
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    main.load_state_dict(state)
    mtp = MtpModel(main, n_stages).eval()
    stages = {n[len("mtp.") :]: convert(n, v) for n, v in params.items() if n.startswith("mtp.")}
    result = mtp.load_state_dict(stages, strict=False)
    # Only the tensors shared with the main model are not in the kit's mtp.* entries.
    assert set(result.missing_keys) == {"embed_tokens.weight", "shared_head.weight"}
    assert not result.unexpected_keys and mtp.use_post_norm
    return main, mtp, config


def compare_mtp(length: int, n_stages: int, seed: int = 0) -> dict[str, float]:
    import torch
    from transformers.cache_utils import MtpCache

    cfg = DecoderConfig(n_layers=2)
    params = mtp_params(cfg, n_stages, seed)
    main, mtp, config = tiny_llama_with_mtp(cfg, params, n_stages)
    tokens = np.random.default_rng(seed).integers(0, cfg.vocab_size, length)
    with torch.no_grad():
        out = main(torch.from_numpy(tokens)[None], output_hidden_states=True)
        normed = main.model(torch.from_numpy(tokens)[None]).last_hidden_state
    # What the generator hands the module (`hidden_states[-1]`) is the final-normed state.
    assert torch.equal(out.hidden_states[-1], normed)
    hidden = decoder_hidden_ref(cfg, params, tokens)
    diffs = {
        "main_logits": float(np.abs(out.logits[0].numpy() - decoder_forward_ref(cfg, params, tokens)).max()),
        "main_hidden": float(np.abs(normed[0].numpy() - hidden).max()),
    }  # fmt: skip
    next_token = int(out.logits[0, -1].argmax())
    # The module's inputs run one token ahead: ids and positions shifted by one.
    ids = torch.from_numpy(np.concatenate((tokens[1:], [next_token])))[None]
    positions = torch.arange(1, length + 1)[None]
    mine_tokens, mine_logits = mtp_draft_ref(cfg, params, tokens, next_token, hidden, n_stages)
    diffs["draft_logits"] = 0.0
    for cache in (None, MtpCache(config=config.get_mtp_config())):
        with torch.no_grad():
            drafted, logits, _ = mtp(ids, torch.from_numpy(hidden)[None], None, positions, cache)
        assert drafted[0].tolist() == mine_tokens.tolist()
        diffs["draft_logits"] = max(
            diffs["draft_logits"], float(np.abs(logits[0].numpy() - mine_logits).max())
        )
    diffs["logit_scale"] = float(np.abs(mine_logits).max())
    return diffs


def test_mtp_chain_matches_transformers_mtp_model():
    torch_and_transformers()
    for length, n_stages, seed in [(1, 1, 0), (5, 2, 1), (12, 3, 2), (20, 3, 3)]:
        diffs = compare_mtp(length, n_stages, seed)
        assert diffs["draft_logits"] < 1e-4 and diffs["main_logits"] < 1e-4, diffs
        assert diffs["main_hidden"] < 1e-5 and diffs["logit_scale"] > 3.0, diffs


def test_mtp_layer_concatenates_the_embedding_first():
    torch = torch_and_transformers()
    cfg = DecoderConfig(n_layers=1)
    params = mtp_params(cfg, 1, seed=4)
    _, mtp, _ = tiny_llama_with_mtp(cfg, params, 1)
    tokens = np.array([3, 9, 27, 40])
    hidden = decoder_hidden_ref(cfg, params, tokens)
    _, mine = mtp_draft_ref(cfg, params, tokens, 5, hidden, 1)
    ids, positions = torch.tensor([[9, 27, 40, 5]]), torch.arange(1, 5)[None]
    with torch.no_grad():
        _, theirs, _ = mtp(ids, torch.from_numpy(hidden)[None], None, positions, None)
        mtp.config.mtp_hidden_states_first = True  # the order of the DeepSeek-V3 paper
        _, paper_order, _ = mtp(ids, torch.from_numpy(hidden)[None], None, positions, None)
    assert np.abs(theirs[0].numpy() - mine).max() < 1e-4
    assert np.abs(paper_order[0].numpy() - mine).max() > 1e-1  # the kit follows the default


# -- eb_accept against the DiffusionGemma sampler -----------------------------------------------


def compare_eb_accept(seed: int, bound: float, shape=(3, 16, 40)) -> dict[str, float]:
    import torch
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
    )

    rng = np.random.default_rng(seed)
    batch, length, vocab = shape
    # Positions from nearly certain to nearly uniform, so every bound cuts somewhere.
    logits = (rng.standard_normal(shape) * rng.uniform(0.2, 6.0, (batch, length, 1))).astype(np.float32)  # fmt: skip
    current = rng.integers(0, vocab, (batch, length))
    sampled = rng.integers(0, vocab, (batch, length))
    sampler = EntropyBoundSampler(EntropyBoundSamplerConfig(entropy_bound=bound), length, vocab, 8)
    accepted = sampler.accept_canvas(
        torch.from_numpy(current), torch.from_numpy(sampled), torch.from_numpy(logits), 3
    )
    theirs = sampler.accepted_token_mask.numpy()
    mine = ops.eb_accept(logits, bound)
    np.testing.assert_array_equal(accepted.numpy(), np.where(mine, sampled, current))
    return {"mismatches": float((theirs != mine).sum()), "accepted": float(mine.sum())}


def test_eb_accept_matches_diffusion_gemma_sampler():
    torch_and_transformers()
    accepted = set()
    for seed in range(4):
        for bound in (1e-6, 0.05, 0.3, 1.0, 3.0, 10.0, 1e3):
            diffs = compare_eb_accept(seed, bound)
            assert diffs["mismatches"] == 0, (seed, bound)
            accepted.add(diffs["accepted"])
    assert min(accepted) == 3 and max(accepted) == 48  # from one per row to everything


def test_eb_accept_in_each_framework_matches_diffusion_gemma_sampler():
    torch = torch_and_transformers()
    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
    )

    from learnai.judge import frameworks

    rng = np.random.default_rng(9)
    logits = (rng.standard_normal((12, 24)) * rng.uniform(0.2, 6.0, (12, 1))).astype(np.float32)
    sampler = EntropyBoundSampler(EntropyBoundSamplerConfig(entropy_bound=0.8), 12, 24, 8)
    tokens = torch.zeros(1, 12, dtype=torch.long)
    sampler.accept_canvas(tokens, tokens, torch.from_numpy(logits)[None], 1)
    want = sampler.accepted_token_mask[0].numpy()
    for name in ("torch", "jax", "mlx"):
        try:
            fw = frameworks.get(name)
        except frameworks.FrameworkUnavailable:
            continue
        np.testing.assert_array_equal(fw.numpy(ops.eb_accept(fw.tensor(logits), 0.8)), want)


if __name__ == "__main__":  # print the measured differences (see the module docstring)
    for name, cfg in GDN_CASES.items():
        chunked = {steps: compare_gdn_chunked(cfg, steps) for steps in (1, 5, 64, 70, 150)}
        print(f"gdn {name} chunked path, x (2, T, {cfg.d_model}):", chunked)
        print(f"gdn {name} prefill 70 + 12 single tokens:", compare_gdn_cached(cfg, 70, 12))
    for name, cfg in HYBRID_CASES.items():
        for steps in (6, 70, 150):
            print(f"hybrid {name} T={steps} vs transformers:", compare_hybrid(cfg, steps))
    for cfg in (HybridConfig(), HybridConfig(n_layers=8, partial_rotary_factor=0.5, rope_base=1e7)):
        for steps in (6, 70):
            print(f"hybrid {cfg.n_layers} layers T={steps} vs mlx-lm:", compare_hybrid_mlx_lm(cfg, steps))  # fmt: skip
    for seed, layer in [(1, 0), (2, 2), (3, 3)]:
        print(f"indexer seed {seed} layer {layer}, T=24:", compare_indexer(seed, 24, layer))
    for length, n_stages, seed in [(1, 1, 0), (5, 2, 1), (12, 3, 2), (20, 3, 3)]:
        print(f"mtp T={length} stages={n_stages}:", compare_mtp(length, n_stages, seed))
    total = sum(compare_eb_accept(s, b)["mismatches"] for s in range(4) for b in (1e-6, 0.05, 0.3, 1.0, 3.0, 10.0, 1e3))  # fmt: skip
    print(
        "eb_accept mismatching positions over 28 (seed, bound) pairs of (3, 16, 40) logits:", total
    )
