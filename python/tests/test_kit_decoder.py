"""DecoderConfig, decoder_params, layer_params, decoder_forward_ref, decoder_block_ref."""

import numpy as np
import pytest

from whetstone.kit import (
    DecoderConfig,
    decoder_block_ref,
    decoder_forward_ref,
    decoder_params,
    layer_params,
    ops,
)

CFG = DecoderConfig()
TOKENS = [3, 17, 42, 5, 9, 60]


def test_config_defaults_and_validation():
    assert (CFG.vocab_size, CFG.d_model, CFG.n_layers, CFG.n_heads, CFG.n_kv_heads) == (
        64, 32, 2, 4, 2,
    )  # fmt: skip
    assert CFG.head_dim == 8 and CFG.d_mlp == 48 and CFG.tie_embeddings
    assert CFG.rope_base == 10000.0 and CFG.norm_eps == 1e-5
    with pytest.raises(ValueError, match="divisible"):
        DecoderConfig(d_model=30, n_heads=4)
    with pytest.raises(ValueError, match="even"):
        DecoderConfig(d_model=12, n_heads=4)
    with pytest.raises(ValueError, match="multiple"):
        DecoderConfig(n_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError, match="n_layers"):
        DecoderConfig(n_layers=0)
    with pytest.raises(AttributeError):
        CFG.n_layers = 3  # frozen


def test_params_have_checkpoint_names_shapes_and_order():
    cfg = DecoderConfig(vocab_size=50, d_model=24, n_layers=2, n_heads=6, n_kv_heads=2, d_mlp=40)
    params = decoder_params(cfg, seed=1)
    shapes = {name: value.shape for name, value in params.items()}
    layer = {
        "input_layernorm.weight": (24,),
        "self_attn.q_proj.weight": (24, 24),
        "self_attn.k_proj.weight": (8, 24),
        "self_attn.v_proj.weight": (8, 24),
        "self_attn.o_proj.weight": (24, 24),
        "post_attention_layernorm.weight": (24,),
        "mlp.gate_proj.weight": (40, 24),
        "mlp.up_proj.weight": (40, 24),
        "mlp.down_proj.weight": (24, 40),
    }
    want = {"model.embed_tokens.weight": (50, 24)}
    for i in range(2):
        want.update({f"model.layers.{i}.{name}": shape for name, shape in layer.items()})
    want["model.norm.weight"] = (24,)
    assert shapes == want
    assert list(shapes) == list(want)  # checkpoint order
    assert all(value.dtype == np.float32 for value in params.values())
    assert "lm_head.weight" not in params  # tied: the head is the embedding

    untied = decoder_params(DecoderConfig(tie_embeddings=False), seed=1)
    assert untied["lm_head.weight"].shape == (64, 32)
    assert list(untied)[-1] == "lm_head.weight"


def test_params_are_seeded_and_norm_gains_are_not_ones():
    a, b, c = decoder_params(CFG, 3), decoder_params(CFG, 3), decoder_params(CFG, 4)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert not np.array_equal(a["model.embed_tokens.weight"], c["model.embed_tokens.weight"])
    gains = [value for name, value in a.items() if name.endswith("norm.weight")]
    assert len(gains) == 2 * CFG.n_layers + 1
    for gain in gains:
        assert 0.5 <= gain.min() and gain.max() <= 1.5 and gain.std() > 0.1
    assert not np.array_equal(gains[0], gains[1])


def test_layer_params_strips_the_prefix():
    params = decoder_params(CFG, 0)
    block = layer_params(params, 1)
    assert len(block) == 9
    assert block["mlp.up_proj.weight"] is params["model.layers.1.mlp.up_proj.weight"]
    with pytest.raises(KeyError):
        layer_params(params, 2)


def test_forward_shapes_dtype_and_determinism():
    params = decoder_params(CFG, 0)
    single = decoder_forward_ref(CFG, params, TOKENS)
    assert single.shape == (6, 64) and single.dtype == np.float32
    assert np.isfinite(single).all() and 1.0 < single.std() < 4.0
    batch = decoder_forward_ref(CFG, params, np.array([TOKENS, TOKENS[::-1]]))
    assert batch.shape == (2, 6, 64)
    np.testing.assert_allclose(batch[0], single, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(single, decoder_forward_ref(CFG, params, np.array(TOKENS)))
    assert decoder_forward_ref(CFG, params, [7]).shape == (1, 64)


def test_forward_is_causal_and_position_aware():
    params = decoder_params(CFG, 2)
    base = decoder_forward_ref(CFG, params, TOKENS)
    changed = decoder_forward_ref(CFG, params, [*TOKENS[:4], 11, TOKENS[5]])
    np.testing.assert_array_equal(changed[:4], base[:4])
    assert not np.allclose(changed[4:], base[4:], atol=1e-3)
    # Order matters: without positions the last row would see the same set of tokens.
    one = decoder_forward_ref(CFG, params, [5, 9, 7])
    other = decoder_forward_ref(CFG, params, [9, 5, 7])
    assert not np.allclose(one[2], other[2], atol=1e-3)


def test_forward_matches_a_from_scratch_implementation():
    cfg = DecoderConfig(vocab_size=20, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_mlp=24,
                        rope_base=100.0)  # fmt: skip
    p = decoder_params(cfg, 5)
    tokens = [1, 4, 4, 19, 0]
    seq = len(tokens)

    def norm(x, gain):
        return x / np.sqrt((x**2).mean(-1, keepdims=True) + cfg.norm_eps) * gain

    def rotate(x, pos):  # x: (head_dim,)
        out = np.empty_like(x)
        for i in range(cfg.head_dim // 2):
            angle = pos * cfg.rope_base ** (-2 * i / cfg.head_dim)
            a, b = x[2 * i], x[2 * i + 1]
            out[2 * i] = a * np.cos(angle) - b * np.sin(angle)
            out[2 * i + 1] = a * np.sin(angle) + b * np.cos(angle)
        return out

    x = p["model.embed_tokens.weight"][tokens].astype(np.float64)
    for layer in range(cfg.n_layers):
        w = {k: v.astype(np.float64) for k, v in layer_params(p, layer).items()}
        h = norm(x, w["input_layernorm.weight"])
        q = (h @ w["self_attn.q_proj.weight"].T).reshape(seq, cfg.n_heads, cfg.head_dim)
        k = (h @ w["self_attn.k_proj.weight"].T).reshape(seq, cfg.n_kv_heads, cfg.head_dim)
        v = (h @ w["self_attn.v_proj.weight"].T).reshape(seq, cfg.n_kv_heads, cfg.head_dim)
        out = np.zeros((seq, cfg.n_heads, cfg.head_dim))
        for t in range(seq):
            for head in range(cfg.n_heads):
                kv = head // (cfg.n_heads // cfg.n_kv_heads)
                scores = np.array([
                    rotate(q[t, head], t) @ rotate(k[s, kv], s) / np.sqrt(cfg.head_dim)
                    for s in range(t + 1)
                ])  # fmt: skip
                weights = np.exp(scores - scores.max())
                weights /= weights.sum()
                out[t, head] = weights @ v[: t + 1, kv]
        x = x + out.reshape(seq, cfg.d_model) @ w["self_attn.o_proj.weight"].T
        h = norm(x, w["post_attention_layernorm.weight"])
        gate = h @ w["mlp.gate_proj.weight"].T
        x = x + (gate / (1 + np.exp(-gate)) * (h @ w["mlp.up_proj.weight"].T)) @ w[
            "mlp.down_proj.weight"
        ].T
    want = norm(x, p["model.norm.weight"]) @ p["model.embed_tokens.weight"].T.astype(np.float64)
    np.testing.assert_allclose(decoder_forward_ref(cfg, p, tokens), want, rtol=1e-4, atol=1e-4)


def test_untied_head_is_used_when_configured():
    cfg = DecoderConfig(tie_embeddings=False)
    params = decoder_params(cfg, 0)
    logits = decoder_forward_ref(cfg, params, TOKENS)
    swapped = dict(params)
    swapped["lm_head.weight"] = params["model.embed_tokens.weight"]
    assert not np.allclose(decoder_forward_ref(cfg, swapped, TOKENS), logits, atol=1e-3)


def test_block_ref_composes_into_the_forward_pass_and_takes_masks_and_positions():
    params = decoder_params(CFG, 6)
    x = params["model.embed_tokens.weight"][np.array([TOKENS])]
    causal = np.tril(np.ones((6, 6), dtype=bool))
    h = x
    for i in range(CFG.n_layers):
        h = decoder_block_ref(CFG, layer_params(params, i), h, causal)
        assert h.shape == x.shape and h.dtype == np.float32
    logits = ops.rms_norm(h, params["model.norm.weight"]) @ params["model.embed_tokens.weight"].T
    np.testing.assert_allclose(logits[0], decoder_forward_ref(CFG, params, TOKENS), rtol=1e-5,
                               atol=1e-5)  # fmt: skip
    block = layer_params(params, 0)
    additive = np.where(causal, 0.0, -np.inf).astype(np.float32)
    np.testing.assert_allclose(decoder_block_ref(CFG, block, x, additive),
                               decoder_block_ref(CFG, block, x, causal), rtol=1e-5, atol=1e-6)  # fmt: skip
    # No mask means no mask: the first position then sees the future.
    assert not np.allclose(decoder_block_ref(CFG, block, x)[0, 0],
                           decoder_block_ref(CFG, block, x, causal)[0, 0], atol=1e-3)  # fmt: skip
    # Positions shift the rotation; a common offset changes nothing (RoPE is relative).
    shifted = decoder_block_ref(CFG, block, x, causal, positions=np.arange(6) + 3)
    np.testing.assert_allclose(shifted, decoder_block_ref(CFG, block, x, causal), rtol=1e-3,
                               atol=1e-4)  # fmt: skip
    stretched = decoder_block_ref(CFG, block, x, causal, positions=np.arange(6) * 2)
    assert not np.allclose(stretched, shifted, atol=1e-3)
    per_row = decoder_block_ref(CFG, block, x, causal, positions=np.arange(6)[None])
    np.testing.assert_allclose(per_row, decoder_block_ref(CFG, block, x, causal), rtol=1e-5,
                               atol=1e-6)  # fmt: skip


@pytest.mark.parametrize("tokens", [[], [[]], [64], [-1], [0.5], [[[1]]]])
def test_forward_rejects_bad_tokens(tokens):
    with pytest.raises(ValueError):
        decoder_forward_ref(CFG, decoder_params(CFG, 0), tokens)


def test_params_convert_to_every_framework_and_ops_reproduce_the_oracle(fw):
    if fw.name == "python":
        pytest.skip("tensor frameworks only")
    params = decoder_params(CFG, 0)
    tensors = fw.to_framework(params)
    assert set(tensors) == set(params) and all(fw.is_tensor(t) for t in tensors.values())
    # One block, written with ops in the solution's framework, as a learner would.
    block = layer_params(tensors, 0)
    x = fw.tensor(params["model.embed_tokens.weight"][np.array([TOKENS])])
    xp, seq = fw.xp, len(TOKENS)
    h = ops.rms_norm(x, block["input_layernorm.weight"], CFG.norm_eps)

    def heads(t, n):
        t = t.reshape(1, seq, n, CFG.head_dim)
        return t.permute(0, 2, 1, 3) if fw.name == "torch" else t.transpose(0, 2, 1, 3)

    positions = fw.tensor(np.arange(seq))
    q = ops.rope(heads(h @ block["self_attn.q_proj.weight"].T, CFG.n_heads), positions)
    k = ops.rope(heads(h @ block["self_attn.k_proj.weight"].T, CFG.n_kv_heads), positions)
    v = heads(h @ block["self_attn.v_proj.weight"].T, CFG.n_kv_heads)
    mask = fw.tensor(np.tril(np.ones((seq, seq), dtype=bool)))
    a = ops.attention(q, k, v, mask)
    a = (a.permute(0, 2, 1, 3) if fw.name == "torch" else a.transpose(0, 2, 1, 3)).reshape(
        1, seq, CFG.d_model
    )
    x = x + a @ block["self_attn.o_proj.weight"].T
    h = ops.rms_norm(x, block["post_attention_layernorm.weight"], CFG.norm_eps)
    x = x + ops.swiglu(h, block["mlp.gate_proj.weight"], block["mlp.up_proj.weight"],
                       block["mlp.down_proj.weight"])  # fmt: skip
    assert xp is not None
    want = decoder_block_ref(
        CFG, layer_params(params, 0), params["model.embed_tokens.weight"][np.array([TOKENS])],
        np.tril(np.ones((seq, seq), dtype=bool)),
    )  # fmt: skip
    np.testing.assert_allclose(fw.numpy(x), want, rtol=1e-4, atol=1e-5)
