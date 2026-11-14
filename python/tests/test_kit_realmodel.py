"""Real-model kit objects: the safetensors reader, the byte-level BPE, the loaders, and
the reference forward pass.

Tests that need downloaded weights skip when the asset is absent
(``ws assets fetch granite-4.0-350m``); the rest run on synthetic files."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from conftest import FIXTURES, REPO_ROOT

from whetstone import assets
from whetstone.judge import frameworks
from whetstone.kit import (
    LARGE_8BIT,
    SMALL,
    ByteLevelBPE,
    RealConfig,
    RealModel,
    SafetensorsFile,
    bf16_to_f32,
    dequantize,
    unpack_codes,
)
from whetstone.kit.bpe import GPT2_PATTERN, byte_alphabet, translate_pattern
from whetstone.kit.realmodel import asset_toml

REFERENCE = FIXTURES / "realmodel" / "granite-4.0-350m.npz"


def asset_dir(spec: assets.AssetSpec) -> Path:
    """The directory of a house model, or skip the test when it is not downloaded."""
    state = assets.state_of(spec, assets.models_dir(REPO_ROOT))
    if not state.present:
        pytest.skip(f"asset not downloaded: {spec.fetch_command}")
    return state.path


@pytest.fixture(scope="module")
def small() -> RealModel:
    return RealModel(asset_dir(SMALL))


@pytest.fixture(scope="module")
def large() -> RealModel:
    return RealModel(asset_dir(LARGE_8BIT))


# -- safetensors ------------------------------------------------------------------------------


def write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]], metadata=None):
    header: dict = {} if metadata is None else {"__metadata__": metadata}
    blob = b""
    for name, (dtype, array) in tensors.items():
        data = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(blob), len(blob) + len(data)],
        }
        blob += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)
    return path


def test_bf16_decoding_is_exact():
    bits = np.array([0x3F80, 0xC020, 0x4049, 0x7F80, 0x0000, 0x8000], dtype=np.uint16)
    want = np.array([1.0, -2.5, 3.140625, np.inf, 0.0, -0.0], dtype=np.float32)
    got = bf16_to_f32(bits)
    assert got.dtype == np.float32 and np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_safetensors_reader_maps_and_decodes(tmp_path):
    values = np.array([[1.0, -2.5, 0.5], [3.0, 0.25, -8.0]], dtype=np.float32)
    bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)
    path = write_safetensors(
        tmp_path / "model.safetensors",
        {
            "w": ("BF16", bf16),
            "codes": ("U32", np.arange(6, dtype=np.uint32).reshape(2, 3)),
            "half": ("F16", values.astype(np.float16)),
            "scalar": ("F32", np.array(7.0, dtype=np.float32)),
        },
        metadata={"format": "pt"},
    )
    st = SafetensorsFile(path)
    assert st.names() == ["w", "codes", "half", "scalar"] and len(st) == 4 and "w" in st
    assert st.metadata == {"format": "pt"} and "__metadata__" not in st.header
    assert st.info("w") == {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}
    raw = st.raw("w")
    assert raw.dtype == np.uint16 and not raw.flags.writeable and not raw.flags.owndata
    assert np.array_equal(raw, bf16)
    assert np.array_equal(st.array("w"), values) and st.array("w").dtype == np.float32
    assert st.array("codes").dtype == np.uint32 and st.array("half").dtype == np.float32
    assert st.array("scalar").shape == () and float(st.array("scalar")) == 7.0
    assert st.n_parameters() == 6 + 6 + 6 + 1 and st.nbytes == path.stat().st_size
    with pytest.raises(KeyError, match="has no tensor 'missing'"):
        st.raw("missing")


def test_safetensors_reader_refuses_garbage(tmp_path):
    short = tmp_path / "short.safetensors"
    short.write_bytes(b"abc")
    with pytest.raises(ValueError, match="too short"):
        SafetensorsFile(short)
    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    with pytest.raises(ValueError, match="implausible header length"):
        SafetensorsFile(huge)


# -- quantized layout --------------------------------------------------------------------------


def test_unpack_and_dequantize_follow_the_mlx_layout():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 256, size=(3, 128), dtype=np.uint8)
    words = codes.view("<u4")  # four codes per word, lowest byte first
    assert words.shape == (3, 32)
    assert np.array_equal(unpack_codes(words), codes)
    scales = rng.uniform(0.001, 0.01, size=(3, 2)).astype(np.float32)
    biases = rng.uniform(-1, 0, size=(3, 2)).astype(np.float32)
    want = codes.astype(np.float32) * np.repeat(scales, 64, axis=1) + np.repeat(biases, 64, axis=1)
    assert np.allclose(dequantize(words, scales, biases, group_size=64, bits=8), want, atol=1e-7)
    # 4-bit codes: eight per word, lowest nibble first.
    nibbles = rng.integers(0, 16, size=(2, 64), dtype=np.uint32)
    packed = (nibbles.reshape(2, 8, 8) << (4 * np.arange(8, dtype=np.uint32))).sum(-1)
    assert np.array_equal(unpack_codes(packed.astype(np.uint32), bits=4), nibbles)


def test_dequantize_equals_mlx():
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(1)
    weight = mx.array(rng.standard_normal((8, 128)).astype(np.float32))
    for bits in (8, 4):
        words, scales, biases = mx.quantize(weight, group_size=64, bits=bits)
        want = np.array(mx.dequantize(words, scales, biases, group_size=64, bits=bits))
        got = dequantize(np.array(words), np.array(scales), np.array(biases), 64, bits)
        assert np.allclose(got, want, atol=1e-6)


# -- the tokenizer ------------------------------------------------------------------------------


def toy_tokenizer_json(pre_tokenizer: dict) -> dict:
    alphabet = byte_alphabet()
    vocab = {char: index for index, char in enumerate(alphabet)}
    merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("Ġ", "w"), ("hell", "o")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    added = [
        {"id": len(vocab), "content": "<|end|>", "special": True},
        {"id": len(vocab) + 1, "content": "<think>", "special": False},
        {"id": len(vocab) + 2, "content": "<think>!", "special": False},
    ]
    return {
        "normalizer": None,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": None,
        "model": {"type": "BPE", "vocab": vocab, "merges": [f"{a} {b}" for a, b in merges]},
        "added_tokens": [
            {"lstrip": False, "rstrip": False, "single_word": False, "normalized": False, **t}
            for t in added
        ],
    }


CL100K = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {
                "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
                " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
            },
            "behavior": "Removed",
            "invert": True,
        },
        {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
    ],
}
GPT2 = {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True}


def test_byte_alphabet_is_the_gpt2_table():
    alphabet = byte_alphabet()
    assert len(set(alphabet)) == 256
    assert alphabet[ord("a")] == "a" and alphabet[ord(" ")] == "Ġ" and alphabet[ord("\n")] == "Ċ"


def test_bpe_merges_by_rank_and_round_trips():
    tok = ByteLevelBPE.from_json(toy_tokenizer_json(CL100K))
    assert tok.vocab_size == 256 + 5 + 3
    ids = tok.encode("hello world")
    assert [tok.token_bytes(i) for i in ids] == [b"hello", b" w", b"o", b"r", b"l", b"d"]
    for text in ["hello world", "", "  spaces  ", "héllo ✓ 東京 🗡️", "tabs\tand\nlines\r\n", "it's 12345"]:
        assert tok.decode(tok.encode(text)) == text
    # A multi-byte character is several single-byte tokens here; decode joins the bytes.
    snowman = tok.encode("☃")
    assert len(snowman) == 3 and tok.decode(snowman[:2]) == "�"


def test_added_tokens_are_cut_out_first_longest_first():
    tok = ByteLevelBPE.from_json(toy_tokenizer_json(CL100K))
    end, think, think_bang = tok.added["<|end|>"], tok.added["<think>"], tok.added["<think>!"]
    assert tok.specials == {"<|end|>": end}
    assert tok.encode("<think>!<think><|end|>") == [think_bang, think, end]
    ids = tok.encode("hello<|end|>hello")
    assert ids == [tok.vocab["hello"], end, tok.vocab["hello"]]
    assert tok.decode(ids) == "hello<|end|>hello"
    assert tok.decode(ids, skip_special_tokens=True) == "hellohello"
    assert tok.token_bytes(end) == b"<|end|>"
    with pytest.raises(KeyError):
        tok.token_bytes(10**6)


def test_the_two_pre_tokenizers_cut_digits_and_spaces_differently():
    """Granite 4.0 (cl100k Split) and Granite 4.2 (GPT-2 ByteLevel) share a vocabulary
    and still disagree: the first cuts digit runs in threes and never attaches a
    leading space to digits."""
    cl100k = translate_pattern(CL100K["pretokenizers"][0]["pattern"]["Regex"])
    gpt2 = translate_pattern(GPT2_PATTERN)
    import re

    assert re.findall(cl100k, "In 2026, 12345") == ["In", " ", "202", "6", ",", " ", "123", "45"]
    assert re.findall(gpt2, "In 2026, 12345") == ["In", " 2026", ",", " 12345"]
    # Unicode classes are exact: superscripts and fractions are numbers, not letters.
    assert re.findall(gpt2, "x²½") == ["x", "²½"]
    # U+001C is whitespace to Python's \s but not Unicode White_Space, which is what
    # the reference uses: there it is an ordinary non-letter and may lead a word.
    assert re.findall(cl100k, "a\x1cb") == ["a", "\x1cb"]
    assert ByteLevelBPE.from_json(toy_tokenizer_json(GPT2)).decode(
        ByteLevelBPE.from_json(toy_tokenizer_json(GPT2)).encode("hello 42")
    ) == "hello 42"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda spec: spec["model"].update(type="WordPiece"), "unsupported tokenizer model"),
        (lambda spec: spec.update(normalizer={"type": "Lowercase"}), "unsupported normalizer"),
        (lambda spec: spec.update(pre_tokenizer={"type": "Whitespace"}), "unsupported pre_tokenizer"),
        (lambda spec: spec["added_tokens"][0].update(lstrip=True), "lstrip/rstrip/single_word"),
        (
            lambda spec: spec.update(
                post_processor={"type": "TemplateProcessing", "single": [], "special_tokens": {"x": 1}}
            ),
            "inserts special tokens",
        ),
    ],
)
def test_unsupported_tokenizer_features_are_refused(change, message):
    spec = toy_tokenizer_json(CL100K)
    change(spec)
    with pytest.raises(ValueError, match=message):
        ByteLevelBPE.from_json(spec)


TEXTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n\treturn n",
    "Whetstone sharpens 🗡️ blades; 東京 is in Japan.",
    "  leading  and   trailing   ",
    "it's can't we'll I'M THEY'RE",
    "1234567890 3.14159 1e-10 ²³ ½ Ⅷ ٣",
    "<|end_of_text|>hello<think>x</think><|im_start|>user\nhi<|im_end|>",
    "<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n",
    "naïve café é é ñ",
    "tab\there\r\nwin\x1cfs\x85nel ls　ideo",
    "한국어 텍스트 עברית العربية हिन्दी 👨‍👩‍👧‍👦 🇫🇷",
    "$$$ !!! ... --- ___ ///  nbsp\u200bzero﻿bom",
    "",
    " ",
    "\n\n\n",
]


@pytest.mark.parametrize("spec", [SMALL, LARGE_8BIT], ids=lambda spec: spec.name)
def test_kit_tokenizer_equals_the_tokenizers_library(spec):
    tokenizers = pytest.importorskip("tokenizers")
    directory = asset_dir(spec)
    reference = tokenizers.Tokenizer.from_file(str(directory / "tokenizer.json"))
    mine = RealModel(directory).tokenizer
    rng = np.random.default_rng(5)
    pool = list("abc XYZ 012 \n\t.,;'\"-_éß東京🙂<|>")
    random_texts = ["".join(rng.choice(pool, size=rng.integers(1, 60))) for _ in range(200)]
    for text in TEXTS + random_texts:
        want = reference.encode(text, add_special_tokens=False).ids
        assert mine.encode(text) == want, text
        assert mine.decode(want) == reference.decode(want, skip_special_tokens=False), text


# -- configs and loaders --------------------------------------------------------------------------


def test_house_models_are_pinned_and_printable_as_toml():
    assert SMALL.repo == "ibm-granite/granite-4.0-350m" and len(SMALL.revision) == 40
    assert LARGE_8BIT.repo == "lmstudio-community/granite-4.2-3b-MLX-8bit"
    block = asset_toml(SMALL)
    import tomllib

    (parsed,), errors = assets.parse_assets(tomllib.loads(block)["assets"])
    assert errors == [] and parsed == SMALL


def test_config_normalization_rejects_what_the_kit_cannot_run():
    base = {
        "model_type": "granite", "num_hidden_layers": 2, "hidden_size": 64,
        "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 128,
        "vocab_size": 100, "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
        "rope_theta": 1e7, "attention_multiplier": 1 / 16, "embedding_multiplier": 1.0,
        "residual_multiplier": 1.0, "logits_scaling": 1.0, "tie_word_embeddings": False,
    }  # fmt: skip
    config = RealConfig.from_json(base)
    assert (config.head_dim, config.fused_mlp, config.quantization) == (16, False, None)
    quantized = RealConfig.from_json({**base, "quantization": {"bits": 8, "group_size": 64}})
    assert quantized.quantization == (8, 64)
    with pytest.raises(ValueError, match="unsupported model_type 'qwen3'"):
        RealConfig.from_json({**base, "model_type": "qwen3"})
    with pytest.raises(ValueError, match="experts or non-attention layers"):
        RealConfig.from_json({**base, "model_type": "granitemoehybrid", "layer_types": ["mamba"]})


def test_small_model_config_tokenizer_and_weights(small):
    cfg = small.config
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim) == (28, 1024, 16, 4, 64)
    assert (cfg.d_ff, cfg.vocab_size, cfg.norm_eps, cfg.rope_theta) == (2048, 100352, 1e-5, 1e7)
    assert cfg.attention_scale == 1 / 64  # 1 / head_dim, not 1 / sqrt(head_dim)
    assert (cfg.embedding_multiplier, cfg.residual_multiplier, cfg.logits_scaling) == (12.0, 0.263, 4.0)
    assert cfg.tied_embeddings and cfg.fused_mlp and cfg.quantization is None
    assert small.config_json["model_type"] == "granitemoehybrid"

    checkpoint = small.checkpoint
    assert len(checkpoint) == 226 and checkpoint.n_parameters() == 352_379_904
    assert "lm_head.weight" not in checkpoint  # tied: the head is the embedding matrix
    assert checkpoint.info("model.layers.0.shared_mlp.input_linear.weight")["shape"] == [4096, 1024]
    assert {info["dtype"] for info in checkpoint.header.values()} == {"BF16"}

    tokenizer = small.tokenizer
    assert tokenizer.encode("The capital of France is") == [791, 6864, 315, 9822, 374]
    assert tokenizer.specials["<|end_of_text|>"] == cfg.eos_id == 100257
    assert "<|start_of_role|>" in tokenizer.specials and "<|im_start|>" not in tokenizer.added
    assert "<|start_of_role|>" in small.chat_template

    weights = small.weights()
    assert len(weights) == 226 and "model.norm.weight" in weights
    gain = weights["model.norm.weight"]
    assert isinstance(gain, np.ndarray) and gain.dtype == np.float32 and not gain.flags.writeable
    assert weights.raw("model.norm.weight").dtype == np.uint16
    assert weights["model.norm.weight"] is gain  # decoded once
    with pytest.raises(KeyError):
        weights["lm_head.weight"]


def test_weights_follow_the_solution_framework(small, fw):
    weights = small.weights().with_framework(fw)
    value = weights["model.layers.0.input_layernorm.weight"]
    if fw.name == "python":
        assert isinstance(value, np.ndarray)
    else:
        assert fw.is_tensor(value) and fw.numpy(value).dtype == np.float32
        assert fw.numpy(value).shape == (1024,)
    # ctx.call converts kit objects the same way.
    assert fw.to_framework((small.weights(),))[0].fw is fw


# -- the reference forward pass ---------------------------------------------------------------------


def test_reference_forward_matches_the_transformers_fixture(small):
    reference = np.load(REFERENCE)
    assert str(reference["revision"]) == SMALL.revision, "regenerate the fixture (make.py)"
    prompts = json.loads((FIXTURES / "realmodel" / "prompts.json").read_text(encoding="utf-8"))
    lm = small.lm()
    columns = reference["columns"]
    worst = 0.0
    for index, prompt in enumerate(prompts):
        ids = small.tokenizer.encode(prompt)
        assert ids == reference[f"ids_{index}"].tolist()
        logits, cache = lm(ids)
        assert logits.shape == (len(ids), 100352) and logits.dtype == np.float32
        assert cache.length == len(ids) and cache.keys[0].shape == (1, 4, len(ids), 64)
        assert np.array_equal(logits.argmax(-1), reference[f"argmax_{index}"])
        worst = max(worst, float(np.abs(logits[:, columns] - reference[f"logits_{index}"]).max()))
        top = reference[f"top_ids_{index}"]
        worst = max(worst, float(np.abs(logits[-1, top] - reference[f"top_logits_{index}"]).max()))
        hidden = lm.hidden_states(ids)
        assert len(hidden) == 29
        for layer in (0, 1, 14):
            want = reference[f"hidden_{index}_{layer}"]
            assert np.allclose(hidden[layer][:, :8], want, rtol=1e-3, atol=1e-3), layer
    # Logits are of magnitude 25; float32 agreement with another implementation.
    assert worst < 5e-4, worst
    assert lm.counters.snapshot() == {"forward_calls": 3, "tokens_processed": 5 + 11 + 18}


def test_cached_and_batched_calls_agree_with_one_full_call(small):
    lm = small.lm()
    ids = small.tokenizer.encode("Whetstone sharpens blades; Tokyo is in Japan.")
    full, _ = lm(ids)
    head, cache = lm(ids[:4])
    rows = [head]
    for token in ids[4:]:
        step, cache = lm([token], cache)
        rows.append(step)
    assert np.abs(np.concatenate(rows) - full).max() < 5e-4
    assert cache.length == len(ids) and not cache.keys[0].flags.writeable
    batch, batch_cache = lm(np.array([ids[:5], ids[5:10]]))
    assert batch.shape == (2, 5, 100352) and batch_cache.batched
    assert np.abs(batch[0] - full[:5]).max() < 5e-4
    with pytest.raises(ValueError, match="token ids must be in"):
        lm([100352])
    with pytest.raises(ValueError, match="different batch sizes"):
        lm(np.array([[1], [2]]), cache)


def test_lm_speaks_the_solution_framework_and_counts_separately(small):
    try:
        torch_fw = frameworks.get("torch")
    except frameworks.FrameworkUnavailable as exc:
        pytest.skip(str(exc))
    lm = small.lm()
    bound = lm.with_framework(torch_fw)
    logits, _ = bound([791, 6864])
    assert torch_fw.is_tensor(logits) and tuple(logits.shape) == (2, 100352)
    assert bound.counters.tokens_processed == 2 and lm.counters.tokens_processed == 0


def test_reference_forward_matches_transformers_on_demand(small):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.AutoModelForCausalLM.from_pretrained(small.path, dtype=torch.float32)
    ids = small.tokenizer.encode("A prompt that is not in the fixture: 3 + 4 =")
    with torch.no_grad():
        want = model.eval()(torch.tensor([ids])).logits[0].numpy()
    got, _ = small.lm()(ids)
    assert np.abs(got - want).max() < 5e-4
    assert np.array_equal(got.argmax(-1), want.argmax(-1))


# -- the large, quantized model ----------------------------------------------------------------------


def test_large_model_layout_is_mlx_8_bit(large):
    cfg = large.config
    assert (cfg.model_type, cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads) == (
        "granite", 40, 2560, 40, 8,
    )  # fmt: skip
    assert cfg.quantization == (8, 64) and not cfg.tied_embeddings and not cfg.fused_mlp
    assert (cfg.embedding_multiplier, cfg.residual_multiplier, cfg.logits_scaling) == (1.0, 1.0, 1.0)
    assert cfg.attention_scale == 1 / 64 and cfg.rope_theta == 1e7

    checkpoint = large.checkpoint
    assert checkpoint.metadata == {"format": "mlx"}
    name = "model.layers.0.self_attn.k_proj"
    assert checkpoint.info(f"{name}.weight") | {"data_offsets": None} == {
        "dtype": "U32", "shape": [512, 640], "data_offsets": None,
    }  # fmt: skip
    assert checkpoint.info(f"{name}.scales")["shape"] == [512, 40]
    assert checkpoint.info(f"{name}.biases")["dtype"] == "BF16"
    # The embedding and the (untied) head are quantized too; the norms are not.
    assert checkpoint.info("model.embed_tokens.weight")["shape"] == [100352, 640]
    assert checkpoint.info("lm_head.weight")["dtype"] == "U32"
    assert checkpoint.info("model.norm.weight") | {"data_offsets": None} == {
        "dtype": "BF16", "shape": [2560], "data_offsets": None,
    }  # fmt: skip

    weight = large.dequantized(name)
    assert weight.shape == (512, 2560) and weight.dtype == np.float32
    assert 0.01 < float(np.abs(weight).max()) < 2.0
    with pytest.raises(ValueError, match="RealLM runs float checkpoints only"):
        large.lm()

    tokenizer = large.tokenizer
    assert tokenizer.encode("<|im_start|>user\nhi<|im_end|>") == [100256, 882, 198, 6151, 100257]
    assert "<|im_start|>assistant" in large.chat_template


def test_large_model_dequantize_equals_mlx(large):
    mx = pytest.importorskip("mlx.core")
    name = "model.layers.3.self_attn.v_proj"
    weights = large.weights(frameworks.get("mlx"))
    words, scales, biases = (weights[f"{name}.{part}"] for part in ("weight", "scales", "biases"))
    assert words.dtype == mx.uint32 and scales.dtype == mx.float32
    want = np.array(mx.dequantize(words, scales, biases, group_size=64, bits=8))
    assert np.array_equal(large.dequantized(name), want)
