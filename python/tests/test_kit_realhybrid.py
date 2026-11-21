"""The real hybrid model (``Qwen/Qwen3.5-0.8B``): the asset, the loader, and the
reference forward pass ``RealHybridLM``.

Tests that need the downloaded weights skip when the asset is absent
(``learnai assets fetch qwen3.5-0.8b``); the rest run on a miniature checkpoint in the
real layout, written here."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURES, REPO_ROOT

from learnai import assets, cli
from learnai.kit import (
    QWEN35,
    HybridConfig,
    RealHybrid,
    RealHybridLM,
    checkpoint_layout,
    hybrid_block_ref,
    hybrid_forward_ref,
    hybrid_params,
    ops,
)
from learnai.kit.realhybrid import (
    TEXT_PREFIX,
    hybrid_config_from_json,
    is_offset_norm,
    kit_name,
)
from learnai.kit.realmodel import _DECODED, KIT_ASSETS, QWEN35_WEIGHTS_FILE, asset_toml

REFERENCE = FIXTURES / "realhybrid" / "qwen3.5-0.8b.npz"
PROMPTS = FIXTURES / "realhybrid" / "prompts.json"
P = TEXT_PREFIX


# -- the asset declaration ---------------------------------------------------------------------


def test_the_asset_is_pinned_and_fetchable_before_a_problem_declares_it(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNAI_ROOT", str(FIXTURES / "m3"))
    monkeypatch.setenv("LEARNAI_MODELS_DIR", str(tmp_path / "models"))
    assert QWEN35.name == "qwen3.5-0.8b" and QWEN35.repo == "Qwen/Qwen3.5-0.8B"
    assert QWEN35.revision == "2fc06364715b967f1860aea9cf38778875588b17"
    assert set(QWEN35.files) >= {"config.json", QWEN35_WEIGHTS_FILE}
    assert KIT_ASSETS[QWEN35.name] is QWEN35 and len(KIT_ASSETS) == 3
    (parsed,), errors = assets.parse_assets(
        [
            {
                "name": QWEN35.name,
                "repo": QWEN35.repo,
                "revision": QWEN35.revision,
                "files": list(QWEN35.files),
                "size_mb": QWEN35.size_mb,
            }
        ]
    )
    assert errors == [] and parsed == QWEN35
    assert f'"{QWEN35_WEIGHTS_FILE}"' in asset_toml(QWEN35)
    seen: list[assets.AssetSpec] = []
    monkeypatch.setattr(assets, "fetch", lambda spec, *args, **kwargs: seen.append(spec))
    assert cli.main(["assets", "fetch", QWEN35.name]) == 0
    assert seen == [QWEN35]


def test_a_problem_that_declares_fewer_files_finds_the_kit_download_present(tmp_path):
    """The kit's spec also fetches tokenizer.json and the index for authors; a problem
    declares only what its cases read, and the same directory satisfies it."""
    directory = tmp_path / QWEN35.name
    directory.mkdir()
    files = {}
    for name in QWEN35.files:
        (directory / name).write_bytes(b"x")
        files[name] = {"size": 1, "sha256": None}
    manifest = {
        "name": QWEN35.name,
        "repo": QWEN35.repo,
        "revision": QWEN35.revision,
        "patterns": list(QWEN35.files),
        "files": files,
    }
    (directory / assets.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    smaller = assets.AssetSpec(
        QWEN35.name, QWEN35.repo, QWEN35.revision, ("config.json", QWEN35_WEIGHTS_FILE), 1750
    )
    assert assets.state_of(smaller, tmp_path).present
    assert assets.state_of(QWEN35, tmp_path).present


# -- the configuration and the layout translation ------------------------------------------------

#: The text settings of the real config.json, reduced to what the forward pass reads.
REAL_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "attn_output_gate": True,
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_size": 1024,
    "intermediate_size": 3584,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_value_head_dim": 128,
    "mlp_only_layers": [],
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": True,
    "vocab_size": 248320,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    },
}


def test_config_is_read_from_the_text_settings():
    cfg = hybrid_config_from_json({"model_type": "qwen3_5", "text_config": REAL_TEXT_CONFIG})
    assert cfg == hybrid_config_from_json(REAL_TEXT_CONFIG)  # the text part alone also works
    assert (cfg.n_layers, cfg.d_model, cfg.d_mlp, cfg.vocab_size) == (24, 1024, 3584, 248320)
    assert (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rotary_dim) == (8, 2, 256, 64)
    assert (cfg.n_key_heads, cfg.n_value_heads, cfg.key_head_dim, cfg.value_head_dim) == (
        16,
        16,
        128,
        128,
    )
    assert (cfg.conv_kernel, cfg.gdn.conv_dim, cfg.norm_eps, cfg.rope_base) == (4, 6144, 1e-6, 1e7)
    assert cfg.tie_embeddings and cfg.layer_types is None
    full = [i for i, kind in enumerate(cfg.layers) if kind == "full_attention"]
    assert full == [3, 7, 11, 15, 19, 23]
    # An explicit list wins over the interval, as in Qwen3_5TextConfig.
    kinds = ["full_attention"] + ["linear_attention"] * 23
    listed = hybrid_config_from_json({**REAL_TEXT_CONFIG, "layer_types": kinds})
    assert listed.layers == tuple(kinds)
    with pytest.raises(ValueError, match="unsupported model_type 'granite'"):
        hybrid_config_from_json({**REAL_TEXT_CONFIG, "model_type": "granite"})
    with pytest.raises(ValueError, match="mlp_only_layers"):
        hybrid_config_from_json({**REAL_TEXT_CONFIG, "mlp_only_layers": [2]})


def test_names_and_offset_norms():
    assert (
        kit_name(P + "layers.3.self_attn.q_norm.weight") == "model.layers.3.self_attn.q_norm.weight"
    )
    assert kit_name(P + "norm.weight") == "model.norm.weight"
    assert kit_name("model.visual.merger.linear_fc1.weight") is None
    assert kit_name("mtp.fc.weight") is None
    offsets = [
        P + "layers.0.input_layernorm.weight",
        P + "layers.0.post_attention_layernorm.weight",
        P + "layers.3.self_attn.q_norm.weight",
        P + "layers.3.self_attn.k_norm.weight",
        P + "norm.weight",
    ]
    assert all(is_offset_norm(name) for name in offsets)
    plain = [
        P + "layers.0.linear_attn.norm.weight",  # the gated norm: a gain, used as stored
        P + "layers.0.linear_attn.A_log",
        P + "embed_tokens.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
    ]
    assert not any(is_offset_norm(name) for name in plain)


def test_checkpoint_layout_renames_and_stores_offsets():
    cfg = HybridConfig()
    params = hybrid_params(cfg, seed=3)
    real = checkpoint_layout(params)
    assert len(real) == len(params) and all(name.startswith(P) for name in real)
    assert np.array_equal(real[P + "embed_tokens.weight"], params["model.embed_tokens.weight"])
    gain = params["model.layers.0.input_layernorm.weight"]
    assert np.allclose(real[P + "layers.0.input_layernorm.weight"] + 1.0, gain, atol=1e-6)
    assert np.allclose(real[P + "norm.weight"] + 1.0, params["model.norm.weight"], atol=1e-6)
    name = "layers.3.self_attn.k_norm.weight"
    assert np.allclose(real[P + name] + 1.0, params["model." + name], atol=1e-6)
    name = "layers.0.linear_attn.norm.weight"
    assert np.array_equal(real[P + name], params["model." + name])
    assert all(value.dtype == np.float32 for value in real.values())
    with pytest.raises(ValueError, match="only a tied hybrid model"):
        checkpoint_layout({"lm_head.weight": np.zeros((2, 2), np.float32)})


# -- a miniature checkpoint in the real layout ------------------------------------------------------


def to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bfloat16 (ties away from zero), as bit patterns."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)


def write_miniature(directory: Path, cfg: HybridConfig, seed: int) -> dict[str, np.ndarray]:
    """A directory shaped like the real asset: the same file names, tensor names, storage
    types (BF16, with ``A_log`` and the gated norm weight F32), offset norms, and unused
    vision and MTP tensors. Returns the kit-layout weights it holds (after BF16 rounding).
    """
    from learnai.kit.safetensors import bf16_to_f32

    real = checkpoint_layout(hybrid_params(cfg, seed))
    real["model.visual.merger.linear_fc1.weight"] = np.ones((3, 2), np.float32)
    real["mtp.norm.weight"] = np.zeros(cfg.d_model, np.float32)
    header: dict = {}
    blob = b""
    held: dict[str, np.ndarray] = {}
    for name, value in real.items():
        is_f32 = name.endswith(("linear_attn.A_log", "linear_attn.norm.weight"))
        stored = value if is_f32 else to_bf16(value)
        data = stored.tobytes()
        header[name] = {
            "dtype": "F32" if is_f32 else "BF16",
            "shape": list(value.shape),
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
        if name.startswith(P):
            decoded = value if is_f32 else bf16_to_f32(stored).reshape(value.shape)
            offset = np.float32(1.0) if is_offset_norm(name) else np.float32(0.0)
            held["model." + name[len(P) :]] = decoded + offset
    encoded = json.dumps(header).encode()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / QWEN35_WEIGHTS_FILE).write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)
    index = {"metadata": {}, "weight_map": dict.fromkeys(header, QWEN35_WEIGHTS_FILE)}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    text = {
        "model_type": "qwen3_5_text",
        "attn_output_gate": True,
        "full_attention_interval": cfg.full_attention_interval,
        "head_dim": cfg.head_dim,
        "hidden_act": "silu",
        "hidden_size": cfg.d_model,
        "intermediate_size": cfg.d_mlp,
        "layer_types": list(cfg.layers),
        "linear_conv_kernel_dim": cfg.conv_kernel,
        "linear_key_head_dim": cfg.key_head_dim,
        "linear_num_key_heads": cfg.n_key_heads,
        "linear_num_value_heads": cfg.n_value_heads,
        "linear_value_head_dim": cfg.value_head_dim,
        "max_position_embeddings": 512,
        "mlp_only_layers": [],
        "num_attention_heads": cfg.n_heads,
        "num_hidden_layers": cfg.n_layers,
        "num_key_value_heads": cfg.n_kv_heads,
        "rms_norm_eps": cfg.norm_eps,
        "tie_word_embeddings": True,
        "vocab_size": cfg.vocab_size,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": cfg.rope_base,
            "partial_rotary_factor": cfg.partial_rotary_factor,
        },
    }
    config = {"model_type": "qwen3_5", "text_config": text, "tie_word_embeddings": True}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return held


@pytest.fixture(scope="module")
def miniature(tmp_path_factory) -> tuple[RealHybrid, HybridConfig, dict[str, np.ndarray]]:
    cfg = HybridConfig(n_layers=5, layer_types=("linear_attention",) * 3 + ("full_attention",) + ("linear_attention",))  # fmt: skip
    directory = tmp_path_factory.mktemp("miniature") / "qwen3.5-0.8b"
    held = write_miniature(directory, cfg, seed=11)
    return RealHybrid(directory), cfg, held


def test_miniature_loader_reads_the_real_layout(miniature):
    model, cfg, held = miniature
    assert model.config == cfg and model.config_json["model_type"] == "qwen3_5"
    names = model.text_names()
    assert len(names) == len(held) == len(model.checkpoint) - 2
    assert "lm_head.weight" not in model.checkpoint
    weights = model.weights()
    stored = weights[P + "layers.0.input_layernorm.weight"]
    assert stored.dtype == np.float32 and abs(float(stored.mean())) < 0.2  # an offset from one
    assert weights.raw(P + "layers.0.linear_attn.A_log").dtype == np.float32
    assert weights.raw(P + "layers.0.linear_attn.dt_bias").dtype == np.uint16
    with pytest.raises(NotImplementedError, match="token ids"):
        model.tokenizer  # noqa: B018
    lm = model.lm()
    assert isinstance(lm, RealHybridLM) and lm.vocab_size == cfg.vocab_size
    block = lm.params(3)
    assert np.array_equal(block["self_attn.q_norm.weight"], held["model.layers.3.self_attn.q_norm.weight"])  # fmt: skip
    assert "linear_attn.conv1d.weight" in lm.params(0) and "self_attn.q_proj.weight" in block


def test_miniature_forward_is_the_kit_hybrid_forward(miniature):
    model, cfg, held = miniature
    lm = model.lm()
    tokens = np.random.default_rng(2).integers(0, cfg.vocab_size, 14)
    want, want_state = hybrid_forward_ref(cfg, held, tokens)
    got, state = lm(tokens)
    assert got.dtype == np.float32 and got.shape == (14, cfg.vocab_size)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
    for kept, expected in zip(state, want_state, strict=True):
        for a, b in zip(kept, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    # Prefill, then one token at a time, through the state.
    rows, state = [], None
    for start, stop in ((0, 6), (6, 10), (10, 11), (11, 12), (12, 13), (13, 14)):
        piece, state = lm(tokens[start:stop], state)
        rows.append(piece)
    np.testing.assert_allclose(np.concatenate(rows), want, rtol=1e-3, atol=1e-4)
    assert lm.counters.snapshot() == {"forward_calls": 7, "tokens_processed": 28}
    # The pieces: embed, block by block, head; and the hidden states they pass along.
    hidden = lm.hidden_states(tokens)
    assert len(hidden) == cfg.n_layers + 1 and hidden[0].shape == (14, cfg.d_model)
    x = lm.embed(tokens)
    assert np.array_equal(x, hidden[0])
    for layer in range(cfg.n_layers):
        x, kept = lm.block(x, layer)
        np.testing.assert_allclose(x, hidden[layer + 1], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(lm.head(x), want, rtol=1e-5, atol=1e-5)
    # The kit-layout oracle for one block gives the same numbers (synthetic cases use it).
    for layer in (0, 3):
        y, kept = hybrid_block_ref(cfg, held, layer, hidden[layer])
        assert np.array_equal(y, hidden[layer + 1])
        again, _ = hybrid_block_ref(cfg, held, layer, hidden[layer][10:], lm.block(hidden[layer][:10], layer)[1])  # fmt: skip
        np.testing.assert_allclose(again, y[10:], rtol=1e-3, atol=1e-4)
    # One block continues from its own state entry.
    first, kept = lm.block(hidden[0][:9], 0)
    rest, _ = lm.block(hidden[0][9:], 0, kept)
    np.testing.assert_allclose(np.concatenate([first, rest]), hidden[1], rtol=1e-3, atol=1e-4)
    # A batch axis.
    batch, _ = lm(np.stack([tokens[:7], tokens[7:]]))
    np.testing.assert_allclose(batch[0], want[:7], rtol=1e-3, atol=1e-4)
    with pytest.raises(ValueError, match="token ids must be in"):
        lm([cfg.vocab_size])
    with pytest.raises(ValueError, match="one entry per layer"):
        lm([1], [None])


def test_miniature_lm_speaks_the_solution_framework(miniature, fw):
    model, cfg, _ = miniature
    lm = model.lm()
    bound = fw.to_framework((lm,), fresh_kit=True)[0]
    logits, state = bound([1, 2, 3])
    if fw.name == "python":
        assert isinstance(logits, np.ndarray)
    else:
        assert fw.is_tensor(logits) and tuple(logits.shape) == (3, cfg.vocab_size)
    assert isinstance(state[0][0], np.ndarray)  # the state stays numpy: it is the oracle's
    assert bound.counters.tokens_processed == 3 and lm.counters.tokens_processed == 0


def qwen_forward(fw, w, cfg, tokens):
    """The port a reference solution of ``real-hybrid-logits`` would write: the raw
    checkpoint names, kit ops, and the framework's elementary functions; one code path
    for torch, JAX, and MLX. Returns the logits and the hidden state after the embedding
    and after every block."""
    xp = fw.xp
    ids = fw.tensor(np.asarray(tokens))
    new = int(ids.shape[0])
    positions = fw.tensor(np.arange(new))
    mask = fw.tensor(np.tril(np.ones((new, new), dtype=bool)))
    eps = cfg.norm_eps

    def norm(x, name):  # the stored weight is an offset from one
        return ops.rms_norm(x, 1.0 + w[name], eps)

    x = w[P + "embed_tokens.weight"][ids]
    hidden = [x]
    for i, kind in enumerate(cfg.layers):
        layer = f"{P}layers.{i}."
        h = norm(x, layer + "input_layernorm.weight")
        if kind == "linear_attention":
            names = ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight",
                     "in_proj_b.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight",
                     "out_proj.weight")  # fmt: skip
            delta = {name: w[layer + "linear_attn." + name] for name in names}
            mixed, _ = ops.gated_delta_layer(h, delta, None, cfg.n_key_heads, eps)
        else:
            attn = layer + "self_attn."
            both = (h @ w[attn + "q_proj.weight"].T).reshape(new, cfg.n_heads, 2 * cfg.head_dim)
            q, gate = both[..., : cfg.head_dim], both[..., cfg.head_dim :]
            k = (h @ w[attn + "k_proj.weight"].T).reshape(new, cfg.n_kv_heads, cfg.head_dim)
            v = (h @ w[attn + "v_proj.weight"].T).reshape(new, cfg.n_kv_heads, cfg.head_dim)
            q = norm(q, attn + "q_norm.weight")
            k = norm(k, attn + "k_norm.weight")
            q, k, v = (xp.swapaxes(t, 0, 1) for t in (q, k, v))
            q = ops.partial_rope(q, positions, cfg.rope_base, cfg.partial_rotary_factor)
            k = ops.partial_rope(k, positions, cfg.rope_base, cfg.partial_rotary_factor)
            attended = xp.swapaxes(ops.attention(q, k, v, mask), 0, 1)
            width = cfg.n_heads * cfg.head_dim
            gate = 1.0 / (1.0 + xp.exp(-gate.reshape(new, width)))
            mixed = (attended.reshape(new, width) * gate) @ w[attn + "o_proj.weight"].T
        x = x + mixed
        h = norm(x, layer + "post_attention_layernorm.weight")
        x = x + ops.swiglu(
            h,
            w[layer + "mlp.gate_proj.weight"],
            w[layer + "mlp.up_proj.weight"],
            w[layer + "mlp.down_proj.weight"],
        )
        hidden.append(x)
    x = norm(x, P + "norm.weight")
    return x @ w[P + "embed_tokens.weight"].T, hidden


def test_miniature_port_from_raw_names_in_every_framework(miniature, fw):
    if fw.name == "python":
        pytest.skip("the port is written for tensor frameworks")
    model, cfg, held = miniature
    tokens = np.random.default_rng(5).integers(0, cfg.vocab_size, 12)
    want, _ = hybrid_forward_ref(cfg, held, tokens)
    weights = fw.to_framework((model.weights(),))[0]  # as ctx.call hands it to a solution
    assert weights.fw is fw and not weights.keep_numpy
    logits, hidden = qwen_forward(fw, weights, cfg, tokens)
    assert fw.is_tensor(logits)
    np.testing.assert_allclose(fw.numpy(logits), want, rtol=1e-3, atol=1e-4)
    reference = model.lm().hidden_states(tokens)
    for got, expected in zip(hidden, reference, strict=True):
        np.testing.assert_allclose(fw.numpy(got), expected, rtol=1e-3, atol=1e-4)
    # The same port on a plain dict in the checkpoint's layout: a synthetic case.
    params = hybrid_params(cfg, seed=4)
    plain = fw.to_framework(checkpoint_layout(params))
    logits, _ = qwen_forward(fw, plain, cfg, tokens)
    want, _ = hybrid_forward_ref(cfg, params, tokens)
    np.testing.assert_allclose(fw.numpy(logits), want, rtol=1e-3, atol=1e-4)
    # Forgetting the `1 +` is far outside any tolerance.
    wrong = dict(plain)
    name = P + "norm.weight"
    wrong[name] = wrong[name] + 1.0
    logits, _ = qwen_forward(fw, wrong, cfg, tokens)
    assert np.abs(fw.numpy(logits) - want).max() > 0.1


def test_framework_weights_are_held_once(tmp_path, fw):
    """Bound to a framework, a tensor is decoded, converted, and the numpy copy dropped:
    the real model's 3 GB are not held twice. The numpy oracle caches what it reads."""
    if fw.name == "python":
        pytest.skip("numpy values are the cache itself")
    cfg = HybridConfig()
    write_miniature(tmp_path / "held-once", cfg, seed=1)
    model = RealHybrid(tmp_path / "held-once")
    name = P + "layers.1.mlp.up_proj.weight"
    key = (str(model.checkpoint.path), name)
    weights = model.weights(fw)
    assert fw.is_tensor(weights[name]) and weights[name] is weights[name]
    assert key not in _DECODED
    model.lm().block(np.zeros((2, cfg.d_model), np.float32), 1)
    assert key in _DECODED
    assert np.array_equal(fw.numpy(model.weights(fw)[name]), _DECODED[key])


def test_text_only_class_of_transformers_loads_the_layout(miniature):
    """``Qwen3_5ForCausalLM.from_pretrained`` on a directory in the real layout: the
    ``model.language_model.`` prefix is mapped, the vision and MTP tensors are ignored,
    the head is tied, and the logits are the kit's."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3_5ForCausalLM

    model, cfg, _ = miniature
    theirs, info = Qwen3_5ForCausalLM.from_pretrained(
        model.path, dtype=torch.float32, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert not info["mismatched_keys"]
    assert theirs.lm_head.weight is theirs.model.embed_tokens.weight
    tokens = np.random.default_rng(9).integers(0, cfg.vocab_size, 16)
    with torch.no_grad():
        result = theirs.eval()(torch.tensor(tokens[None]), output_hidden_states=True)
    lm = model.lm()
    got, _ = lm(tokens)
    assert np.abs(got - result.logits[0].numpy()).max() < 5e-4
    hidden = lm.hidden_states(tokens)
    assert len(result.hidden_states) == len(hidden)
    # transformers applies the final norm to the last entry; the kit does not.
    hidden[-1] = lm.final_norm(hidden[-1])
    for ours, their in zip(hidden, result.hidden_states, strict=True):
        assert np.abs(ours - their[0].numpy()).max() < 5e-4


# -- the real weights ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real() -> RealHybrid:
    state = assets.state_of(QWEN35, assets.models_dir(REPO_ROOT))
    if not state.present:
        pytest.skip(f"asset not downloaded: {QWEN35.fetch_command}")
    return RealHybrid(state.path)


def prompts() -> list[list[int]]:
    return [entry["ids"] for entry in json.loads(PROMPTS.read_text(encoding="utf-8"))]


def test_real_checkpoint_layout(real):
    cfg = real.config
    assert cfg.layer_types is not None  # the file lists the schedule; the rule gives the same
    assert cfg.layers == hybrid_config_from_json(REAL_TEXT_CONFIG).layers
    assert (cfg.n_layers, cfg.d_model, cfg.head_dim, cfg.rotary_dim) == (24, 1024, 256, 64)
    assert real.config_json["text_config"]["layer_types"] == list(cfg.layers)
    checkpoint = real.checkpoint
    assert len(checkpoint) == 488 and checkpoint.n_parameters() == 873_438_784
    assert checkpoint.nbytes == 1_746_942_600 and checkpoint.metadata == {}
    names = checkpoint.names()
    text = real.text_names()
    assert len(text) == 320
    assert sum(name.startswith("model.visual.") for name in names) == 153
    assert sum(name.startswith("mtp.") for name in names) == 15
    assert not any("lm_head" in name for name in names)  # tied: the head is the embedding

    def size(name: str) -> int:
        return int(np.prod(checkpoint.info(name)["shape"]))

    assert sum(size(name) for name in text) == 752_393_024
    # Everything is BF16 except two kinds of small tensor in the delta layers.
    f32 = [name for name in names if checkpoint.info(name)["dtype"] == "F32"]
    assert {checkpoint.info(name)["dtype"] for name in names} == {"BF16", "F32"}
    assert len(f32) == 36 and sum(size(name) for name in f32) == 2592
    assert {name.rsplit("linear_attn.", 1)[1] for name in f32} == {"A_log", "norm.weight"}
    shapes = {
        "embed_tokens.weight": [248320, 1024],
        "norm.weight": [1024],
        "layers.0.input_layernorm.weight": [1024],
        "layers.0.post_attention_layernorm.weight": [1024],
        "layers.0.mlp.gate_proj.weight": [3584, 1024],
        "layers.0.mlp.up_proj.weight": [3584, 1024],
        "layers.0.mlp.down_proj.weight": [1024, 3584],
        "layers.0.linear_attn.in_proj_qkv.weight": [6144, 1024],
        "layers.0.linear_attn.in_proj_z.weight": [2048, 1024],
        "layers.0.linear_attn.in_proj_a.weight": [16, 1024],
        "layers.0.linear_attn.in_proj_b.weight": [16, 1024],
        "layers.0.linear_attn.conv1d.weight": [6144, 1, 4],
        "layers.0.linear_attn.A_log": [16],
        "layers.0.linear_attn.dt_bias": [16],
        "layers.0.linear_attn.norm.weight": [128],
        "layers.0.linear_attn.out_proj.weight": [1024, 2048],
        "layers.3.self_attn.q_proj.weight": [4096, 1024],
        "layers.3.self_attn.k_proj.weight": [512, 1024],
        "layers.3.self_attn.v_proj.weight": [512, 1024],
        "layers.3.self_attn.o_proj.weight": [1024, 2048],
        "layers.3.self_attn.q_norm.weight": [256],
        "layers.3.self_attn.k_norm.weight": [256],
    }
    for name, shape in shapes.items():
        assert checkpoint.info(P + name)["shape"] == shape, name
    for layer, kind in enumerate(cfg.layers):
        has_delta = f"{P}layers.{layer}.linear_attn.A_log" in checkpoint
        has_attention = f"{P}layers.{layer}.self_attn.q_proj.weight" in checkpoint
        assert (has_delta, has_attention) == (kind == "linear_attention", kind == "full_attention")
    # The block norms are stored as offsets (near zero); the gated norm is a plain gain.
    weights = real.weights()
    assert abs(float(weights[P + "layers.0.input_layernorm.weight"].mean())) < 0.5
    assert 0.5 < float(weights[P + "layers.0.linear_attn.norm.weight"].mean()) < 1.5


def test_real_forward_matches_the_transformers_fixture(real):
    reference = np.load(REFERENCE)
    assert str(reference["revision"]) == QWEN35.revision, "regenerate the fixture (make.py)"
    lm = real.lm()
    columns, hidden_columns = reference["columns"], reference["hidden_columns"]
    worst_logits = worst_hidden = 0.0
    for index, ids in enumerate(prompts()):
        assert ids == reference[f"ids_{index}"].tolist() and len(ids) <= 16
        logits, state = lm(ids)
        assert logits.shape == (len(ids), 248320) and logits.dtype == np.float32
        assert state[0][0].shape == (3, 6144) and state[0][1].shape == (16, 128, 128)
        assert state[3][0].shape == (2, len(ids), 256)
        assert np.array_equal(logits.argmax(-1), reference[f"argmax_{index}"])
        diff = np.abs(logits[:, columns] - reference[f"logits_{index}"]).max()
        top = reference[f"top_ids_{index}"]
        diff = max(diff, np.abs(logits[-1, top] - reference[f"top_logits_{index}"]).max())
        worst_logits = max(worst_logits, float(diff))
        hidden = lm.hidden_states(ids)
        assert len(hidden) == 25
        hidden[24] = lm.final_norm(hidden[24])  # as transformers reports its last entry
        for entry in (0, 1, 2, 3, 4, 13, 14, 24):
            want = reference[f"hidden_{index}_{entry}"]
            got = hidden[entry][:, hidden_columns]
            assert np.allclose(got, want, rtol=1e-3, atol=1e-3), (index, entry)
            if entry in (1, 2, 13, 14):  # outputs of linear-attention layers 0, 1, 12, 13
                worst_hidden = max(worst_hidden, float(np.abs(got - want).max()))
    # Measured 2026-11-21: see docs/PROBLEM_AUTHORING.md section 15.
    assert worst_logits < 1e-3, worst_logits
    assert worst_hidden < 1e-3, worst_hidden


def test_real_cached_decoding_agrees_with_one_pass(real):
    lm = real.lm()
    ids = prompts()[1]
    full, _ = lm(ids)
    head, state = lm(ids[:5])
    rows = [head]
    for token in ids[5:]:
        step, state = lm([token], state)
        rows.append(step)
    assert np.abs(np.concatenate(rows) - full).max() < 1e-3


def test_real_port_in_each_framework_matches_the_fixture(real, fw):
    """The from-scratch port (``qwen_forward`` above) on the real weights in torch, JAX,
    and MLX against ``transformers``: one prompt, hidden states and logits."""
    if fw.name == "python":
        pytest.skip("the port is written for tensor frameworks")
    reference = np.load(REFERENCE)
    index = 1
    ids = prompts()[index]
    logits, hidden = qwen_forward(fw, real.weights(fw), real.config, ids)
    logits = fw.numpy(logits)
    assert np.array_equal(logits.argmax(-1), reference[f"argmax_{index}"])
    want = reference[f"logits_{index}"]
    np.testing.assert_allclose(logits[:, reference["columns"]], want, rtol=1e-3, atol=1e-3)
    for entry in (1, 2, 4, 13, 14):
        got = fw.numpy(hidden[entry])[:, reference["hidden_columns"]]
        want = reference[f"hidden_{index}_{entry}"]
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3, err_msg=str(entry))


def test_real_forward_matches_transformers_on_demand(real):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3_5ForCausalLM

    model, info = Qwen3_5ForCausalLM.from_pretrained(
        real.path, dtype=torch.float32, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    ids = [9419, 11, 1814, 0, 220, 18, 478, 220, 19, 283]  # not in the fixture
    with torch.no_grad():
        want = model.eval()(torch.tensor([ids])).logits[0].numpy()
    got, _ = real.lm()(ids)
    assert np.abs(got - want).max() < 1e-3
    assert np.array_equal(got.argmax(-1), want.argmax(-1))
