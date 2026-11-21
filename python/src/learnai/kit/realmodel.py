"""Real open-model weights for real-model problems: config, tokenizer, weights, reference.

    model = RealModel(ctx.asset("granite-4.0-350m"))

    model.config            # RealConfig: the architecture, read from config.json
    model.config_json       # the raw config.json dict
    model.tokenizer         # ByteLevelBPE built from tokenizer.json (kit/bpe.py)
    model.chat_template     # the Jinja chat template text, or None
    model.checkpoint        # SafetensorsFile: memory-mapped (kit/safetensors.py)
    model.weights()         # name -> float32 tensor in the solution's framework, lazily
    model.lm()              # RealLM: the reference forward pass, called like TinyLM

Nothing is read until it is used, and weights are decoded one tensor at a time, so a
test that needs two layers pays for two layers. Decoded tensors are cached per file
for the life of the process, so the cases of one run share them.

The house models (``SMALL``, ``LARGE_8BIT``) are IBM Granite checkpoints. What a
from-scratch forward pass has to do to match their logits is exactly what ``RealLM``
does; the list is in ``RealLM``'s docstring and every number comes from ``RealConfig``.

Only the standard library and numpy are needed here; ``transformers``, ``tokenizers``,
and ``mlx-lm`` (the ``models`` extra) are what authors generate fixtures with.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learnai.assets import AssetSpec
from learnai.kit.bpe import ByteLevelBPE
from learnai.kit.counters import Counters
from learnai.kit.safetensors import SafetensorsFile

_TOKENIZER_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

#: The small house model: 352 M parameters, BF16, 705 MB. Fast enough for CPU tests.
SMALL = AssetSpec(
    name="granite-4.0-350m",
    repo="ibm-granite/granite-4.0-350m",
    revision="bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2",
    files=(*_TOKENIZER_FILES, "special_tokens_map.json", "model.safetensors"),
    size_mb=712,
)
#: The large house model as problems load it: 3.66 B parameters at 8 bits (MLX affine
#: quantization, group size 64), 3.89 GB. MLX-only problems.
LARGE_8BIT = AssetSpec(
    name="granite-4.2-3b-mlx-8bit",
    repo="lmstudio-community/granite-4.2-3b-MLX-8bit",
    revision="7c630b526561db6b4641767c54595fe494176695",
    files=(*_TOKENIZER_FILES, "model.safetensors"),
    size_mb=3896,
)
HOUSE_MODELS = {spec.name: spec for spec in (SMALL, LARGE_8BIT)}

#: The hybrid model of the `real-hybrid-model` problems: `Qwen/Qwen3.5-0.8B`, 873 M
#: stored parameters in BF16 (752 M of them the text decoder), 1.75 GB, Apache-2.0.
#: Not a house model: only those problems use it (see ``kit/realhybrid.py``). The
#: weight file keeps its repository name, ``QWEN35_WEIGHTS_FILE``. A problem declares
#: only ``config.json`` and the weights, because its prompts are token ids; the other
#: two files are for authors: ``transformers`` needs the index to find a weight file
#: with this name, and ``tokenizer.json`` makes fixture prompts from text.
QWEN35_WEIGHTS_FILE = "model.safetensors-00001-of-00001.safetensors"
QWEN35 = AssetSpec(
    name="qwen3.5-0.8b",
    repo="Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
    files=("config.json", "tokenizer.json", "model.safetensors.index.json", QWEN35_WEIGHTS_FILE),
    size_mb=1760,
)
#: Every asset the kit knows by name: ``learnai assets fetch`` accepts these before any
#: problem declares them.
KIT_ASSETS = {**HOUSE_MODELS, QWEN35.name: QWEN35}


def asset_toml(spec: AssetSpec) -> str:
    """The ``[[assets]]`` block for ``spec``, to paste into a problem.toml."""
    files = ", ".join(f'"{name}"' for name in spec.files)
    return (
        "[[assets]]\n"
        f'name = "{spec.name}"\n'
        f'repo = "{spec.repo}"\n'
        f'revision = "{spec.revision}"\n'
        f"files = [{files}]\n"
        f"size_mb = {spec.size_mb:g}\n"
    )


@dataclass(frozen=True)
class RealConfig:
    """The architecture of a checkpoint, normalized from its ``config.json``.

    Every field a forward pass needs is here, so a problem can hand ``cfg`` to a
    solution without the solution parsing JSON. See ``RealLM`` for how each is used.
    """

    model_type: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int  # the SwiGLU hidden width
    vocab_size: int
    max_positions: int
    norm_eps: float
    rope_theta: float
    #: Attention logits are ``(q @ k^T) * attention_scale``. Granite: ``1/64`` for a
    #: head of 64 dimensions, which is ``1/head_dim``, not ``1/sqrt(head_dim)``.
    attention_scale: float
    #: The embedding output is multiplied by this (12 for the small model).
    embedding_multiplier: float
    #: Each branch (attention, MLP) is multiplied by this before it joins the residual.
    residual_multiplier: float
    #: The logits are divided by this (4 for the small model).
    logits_scaling: float
    #: True: the output head is the embedding matrix; there is no ``lm_head.weight``.
    tied_embeddings: bool
    #: True: the SwiGLU gate and up projections are one matrix
    #: (``shared_mlp.input_linear``, gate rows first); False: ``mlp.gate_proj``,
    #: ``mlp.up_proj``, ``mlp.down_proj``.
    fused_mlp: bool
    #: ``(bits, group_size)`` when the linear layers and embeddings are stored
    #: quantized in the MLX layout (see ``dequantize``), else None.
    quantization: tuple[int, int] | None
    bos_id: int | None
    eos_id: int | None
    pad_id: int | None

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> RealConfig:
        kind = raw.get("model_type")
        if kind not in ("granite", "granitemoehybrid"):
            raise ValueError(
                f"unsupported model_type {kind!r}: the kit knows `granite` and "
                "`granitemoehybrid` (see learnai/kit/realmodel.py)"
            )
        # `transformers` 4 writes "attention" for an attention layer; a config saved
        # again by `transformers` 5 says "full_attention". Either a list with one
        # entry per layer or (in hand-written configs) a single string is accepted.
        kinds = raw.get("layer_types") or []
        kinds = {kinds} if isinstance(kinds, str) else set(kinds)
        if kind == "granitemoehybrid" and (
            raw.get("num_local_experts") or kinds - {"attention", "full_attention"}
        ):
            raise ValueError("granitemoehybrid with experts or non-attention layers is unsupported")
        quant = raw.get("quantization")
        if quant is not None and quant.get("mode", "affine") != "affine":
            raise ValueError(f"unsupported quantization mode {quant.get('mode')!r}")
        heads = int(raw["num_attention_heads"])
        rope = raw.get("rope_theta") or (raw.get("rope_parameters") or {}).get("rope_theta")
        return cls(
            model_type=str(kind),
            n_layers=int(raw["num_hidden_layers"]),
            d_model=int(raw["hidden_size"]),
            n_heads=heads,
            n_kv_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["hidden_size"]) // heads,
            d_ff=int(raw["intermediate_size"]),
            vocab_size=int(raw["vocab_size"]),
            max_positions=int(raw["max_position_embeddings"]),
            norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(rope),
            attention_scale=float(raw["attention_multiplier"]),
            embedding_multiplier=float(raw["embedding_multiplier"]),
            residual_multiplier=float(raw["residual_multiplier"]),
            logits_scaling=float(raw["logits_scaling"]),
            tied_embeddings=bool(raw["tie_word_embeddings"]),
            fused_mlp=kind == "granitemoehybrid",
            quantization=(int(quant["bits"]), int(quant["group_size"])) if quant else None,
            bos_id=raw.get("bos_token_id"),
            eos_id=raw.get("eos_token_id"),
            pad_id=raw.get("pad_token_id"),
        )


def unpack_codes(words: np.ndarray, bits: int = 8) -> np.ndarray:
    """Unpack MLX-packed quantization codes: ``(out, in * bits / 32)`` uint32 words
    to ``(out, in)`` uint8 codes.

    Codes are packed little end first: code ``j`` of a word occupies bits
    ``[j * bits, (j + 1) * bits)``. For 8 bits that is simply the word's four bytes in
    memory order, so this is a view, not a copy.
    """
    words = np.ascontiguousarray(words, dtype="<u4")
    if bits == 8:
        return words.view(np.uint8).reshape(words.shape[0], -1)
    if bits not in (2, 4):
        raise ValueError("bits must be 2, 4, or 8")
    per_word = 32 // bits
    shifts = (np.arange(per_word, dtype=np.uint32) * bits)[None, None, :]
    codes = (words[:, :, None] >> shifts) & np.uint32((1 << bits) - 1)
    return codes.reshape(words.shape[0], -1).astype(np.uint8)


def dequantize(
    words: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size: int = 64, bits: int = 8
) -> np.ndarray:
    """Float32 weights from the MLX affine layout; equals ``mx.dequantize``.

    ``words`` is ``<name>.weight`` (packed codes), ``scales`` and ``biases`` are
    ``<name>.scales`` and ``<name>.biases`` with shape ``(out, in / group_size)``. Each
    run of ``group_size`` consecutive codes along the input axis shares one scale and
    one bias: ``w = code * scale + bias``.
    """
    codes = unpack_codes(words, bits)
    out = codes.shape[0]
    grouped = codes.reshape(out, -1, group_size).astype(np.float32)
    scales = np.asarray(scales, dtype=np.float32)[:, :, None]
    biases = np.asarray(biases, dtype=np.float32)[:, :, None]
    return (grouped * scales + biases).reshape(out, -1)


#: Decoded float32 tensors, shared by every RealModel of one file in this process.
_DECODED: dict[tuple[str, str], np.ndarray] = {}


class Weights(Mapping):
    """The tensors of a checkpoint by name, decoded on first use.

    Floating tensors (BF16, F16, F32) come as float32, integer tensors (the packed
    words of a quantized checkpoint) as stored. In a tensor framework the values are
    that framework's tensors; with no framework, or ``python``, they are numpy arrays
    (a hundred-thousand-row embedding as nested lists would help nobody).

    Passing a ``Weights`` through ``ctx.call`` rebinds it to the solution's framework,
    like every kit object. ``weights.raw(name)`` is the zero-copy view of the stored
    bytes (BF16 as uint16), for problems about the file format itself.
    """

    def __init__(
        self, checkpoint: SafetensorsFile, fw: Any = None, *, keep_numpy: bool = True
    ) -> None:
        self.checkpoint = checkpoint
        self.fw = fw
        #: False: a tensor handed out in a framework is decoded, converted, and the
        #: numpy copy dropped, so a large checkpoint is held once and not twice
        #: (``RealHybrid`` asks for this; ``numpy(name)`` still caches what it is asked).
        self.keep_numpy = keep_numpy
        self._tensors: dict[str, Any] = {}

    def with_framework(self, fw: Any) -> Weights:
        return Weights(self.checkpoint, fw, keep_numpy=self.keep_numpy)

    def numpy(self, name: str) -> np.ndarray:
        """The tensor as a read-only numpy array (float32 for floats), cached."""
        key = (str(self.checkpoint.path), name)
        if key not in _DECODED:
            array = self.checkpoint.array(name)
            array.setflags(write=False)
            _DECODED[key] = array
        return _DECODED[key]

    def raw(self, name: str) -> np.ndarray:
        return self.checkpoint.raw(name)

    def __getitem__(self, name: str) -> Any:
        if name not in self.checkpoint:
            raise KeyError(name)
        if self.fw is None or getattr(self.fw, "name", "python") == "python":
            return self.numpy(name)
        if name not in self._tensors:
            decoded = _DECODED.get((str(self.checkpoint.path), name))
            if decoded is None and not self.keep_numpy:
                decoded = self.checkpoint.array(name)
            array = self.numpy(name) if decoded is None else decoded
            # Tensors keep the stored integer width: packed uint32 words stay uint32.
            self._tensors[name] = self.fw.tensor(array, dtype=array.dtype)
        return self._tensors[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.checkpoint.names())

    def __len__(self) -> int:
        return len(self.checkpoint)

    def __repr__(self) -> str:
        fw = getattr(self.fw, "name", "numpy")
        return f"Weights({self.checkpoint.path.name!r}, {len(self)} tensors, {fw})"


class RealModel:
    """A downloaded model directory: config, tokenizer, checkpoint. See the module docs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not (self.path / "config.json").is_file():
            raise FileNotFoundError(f"no config.json in {self.path}")
        self._config_json: dict[str, Any] | None = None
        self._tokenizer: ByteLevelBPE | None = None
        self._checkpoint: SafetensorsFile | None = None

    @property
    def config_json(self) -> dict[str, Any]:
        if self._config_json is None:
            text = (self.path / "config.json").read_text(encoding="utf-8")
            self._config_json = json.loads(text)
        return self._config_json

    @property
    def config(self) -> RealConfig:
        return RealConfig.from_json(self.config_json)

    @property
    def tokenizer_json(self) -> dict[str, Any]:
        """The parsed ``tokenizer.json`` (what a tokenizer problem hands the solution)."""
        return json.loads((self.path / "tokenizer.json").read_text(encoding="utf-8"))

    @property
    def tokenizer(self) -> ByteLevelBPE:
        if self._tokenizer is None:
            self._tokenizer = ByteLevelBPE.from_json(self.tokenizer_json)
        return self._tokenizer

    @property
    def chat_template(self) -> str | None:
        """The Jinja chat template: ``chat_template.jinja``, else tokenizer_config.json."""
        file = self.path / "chat_template.jinja"
        if file.is_file():
            return file.read_text(encoding="utf-8")
        config = self.path / "tokenizer_config.json"
        if config.is_file():
            return json.loads(config.read_text(encoding="utf-8")).get("chat_template")
        return None

    #: The weight file inside the directory (a subclass may name another).
    weights_file = "model.safetensors"

    @property
    def checkpoint(self) -> SafetensorsFile:
        if self._checkpoint is None:
            self._checkpoint = SafetensorsFile(self.path / self.weights_file)
        return self._checkpoint

    def weights(self, fw: Any = None) -> Weights:
        """Name -> tensor, lazily (see ``Weights``). ``fw`` is a framework adapter
        (``ctx.fw``); leave it out when the result goes through ``ctx.call``."""
        return Weights(self.checkpoint, fw)

    def dequantized(self, name: str) -> np.ndarray:
        """Float32 weights of one quantized layer, ``name`` without ``.weight``
        (``"model.layers.0.self_attn.k_proj"``). Only for a quantized checkpoint."""
        quant = self.config.quantization
        if quant is None:
            raise ValueError("this checkpoint is not quantized")
        bits, group_size = quant
        weights = self.weights()
        return dequantize(
            weights.raw(f"{name}.weight"),
            weights.numpy(f"{name}.scales"),
            weights.numpy(f"{name}.biases"),
            group_size,
            bits,
        )

    def lm(self, fw: Any = None) -> RealLM:
        """The reference forward pass over these weights (see ``RealLM``)."""
        return RealLM(self, fw)

    def __repr__(self) -> str:
        return f"RealModel({self.path.name!r})"


@dataclass(frozen=True)
class RealCache:
    """Keys and values of every token seen so far, per layer. Opaque and immutable.

    ``keys[layer]`` and ``values[layer]`` have shape ``(batch, n_kv_heads, length,
    head_dim)``: the cache is at KV-head width, and keys are stored already rotated.
    """

    keys: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]
    length: int
    batched: bool

    def __repr__(self) -> str:
        return f"RealCache(length={self.length}, layers={len(self.keys)})"


def _rms_norm(x: np.ndarray, gain: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps)) * gain


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


class RealLM:
    """The reference forward pass of a house model, called like ``TinyLM``.

        lm = RealModel(ctx.asset("granite-4.0-350m")).lm()
        logits, cache = lm(tokens)              # tokens: (T,) or (B, T) ids
        logits, cache = lm(next_tokens, cache)  # continue the prefix

    ``logits`` has one row per token passed in this call, float32, in the framework
    the model is bound to. ``lm.counters`` counts ``forward_calls`` and
    ``tokens_processed``. numpy inside, so every framework sees identical numbers;
    they agree with ``transformers`` to about 1e-4 absolute on logits of magnitude 25
    (the judge's tests check this when the ``models`` extra is installed).

    What the forward pass does, in order. This is the complete list of what a
    from-scratch implementation needs to match the logits:

    1.  Tokens are embedded with ``model.embed_tokens.weight`` and the result is
        multiplied by ``embedding_multiplier``. No position embedding is added, and
        the tokenizer adds no beginning-of-text token.
    2.  Each of the ``n_layers`` blocks is pre-norm: ``x = x + r * attn(norm1(x))``,
        then ``x = x + r * mlp(norm2(x))``, with ``r = residual_multiplier``.
    3.  The norm is RMSNorm with a learned gain and no bias:
        ``x / sqrt(mean(x^2) + eps) * weight``, ``eps`` inside the square root.
    4.  Attention is grouped-query: ``q_proj`` gives ``n_heads`` heads, ``k_proj`` and
        ``v_proj`` give ``n_kv_heads``; KV head ``j`` serves the ``n_heads /
        n_kv_heads`` consecutive query heads ``j * g`` to ``j * g + g - 1``. No biases.
    5.  RoPE rotates queries and keys (not values) in the half-split layout: dimension
        ``i`` pairs with ``i + head_dim / 2``, the angle of pair ``i`` at position
        ``p`` is ``p * rope_theta ** (-2 i / head_dim)``, and
        ``x' = x * cos + rotate(x) * sin`` with ``rotate([a, b]) = [-b, a]``.
    6.  Attention logits are ``(q @ k^T) * attention_scale`` plus the causal mask;
        the scale is ``1 / head_dim``, not ``1 / sqrt(head_dim)``.
    7.  The MLP is SwiGLU: ``down(silu(gate(x)) * up(x))``. In the small model gate
        and up are one matrix, ``shared_mlp.input_linear`` with ``2 * d_ff`` rows:
        the first ``d_ff`` rows are the gate, the rest are up; ``output_linear`` is
        down. The large model stores ``mlp.gate_proj``, ``up_proj``, ``down_proj``.
    8.  After the last block comes ``model.norm`` (the same RMSNorm), then the head:
        the embedding matrix itself when ``tied_embeddings``, else
        ``lm_head.weight``; the logits are divided by ``logits_scaling``.
    9.  Weights are stored as bfloat16 and are used as float32.

    A quantized checkpoint is not accepted here (its float weights would not fit the
    budget of a test); use ``RealModel.dequantized`` for single layers.
    """

    def __init__(self, model: RealModel, fw: Any = None, counters: Counters | None = None) -> None:
        self.model = model
        self.config = model.config
        if self.config.quantization is not None:
            raise ValueError(
                "RealLM runs float checkpoints only; this one is quantized "
                "(use RealModel.dequantized for individual layers)"
            )
        self.fw = fw
        self.counters = counters or Counters("forward_calls", "tokens_processed")
        self._weights = model.weights()

    def with_framework(self, fw: Any) -> RealLM:
        """The same model bound to another framework, with fresh counters."""
        return RealLM(self.model, fw)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    # -- pieces, exposed so fixtures and oracles can use them one at a time ---------

    def _w(self, name: str) -> np.ndarray:
        return self._weights.numpy(name)

    def embed(self, tokens: np.ndarray) -> np.ndarray:
        """Step 1: ``(..., T)`` ids to ``(..., T, d_model)`` float32."""
        table = self._w("model.embed_tokens.weight")
        return table[np.asarray(tokens)] * np.float32(self.config.embedding_multiplier)

    def _rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Rotate ``(B, heads, T, head_dim)`` at integer ``positions`` of shape ``(T,)``."""
        cfg = self.config
        half = cfg.head_dim // 2
        inverse = cfg.rope_theta ** (-np.arange(half, dtype=np.float64) * 2.0 / cfg.head_dim)
        angles = positions.astype(np.float64)[:, None] * inverse[None, :]
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(np.float32)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(np.float32)
        rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return x * cos + rotated * sin

    def block(
        self,
        x: np.ndarray,
        layer: int,
        positions: np.ndarray | None = None,
        past: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Steps 2 to 7 for one layer: ``(B, T, d_model)`` in and out.

        ``positions`` are the absolute positions of the ``T`` tokens (default
        ``past length + arange(T)``); ``past`` is this layer's ``(keys, values)`` from
        earlier tokens. Returns the new hidden state and the extended keys and values.
        """
        cfg = self.config
        prefix = f"model.layers.{layer}."
        batch, length, _ = x.shape
        n_past = 0 if past is None else past[0].shape[2]
        if positions is None:
            positions = n_past + np.arange(length)
        r = np.float32(cfg.residual_multiplier)

        h = _rms_norm(x, self._w(prefix + "input_layernorm.weight"), cfg.norm_eps)

        def heads(name: str, n: int) -> np.ndarray:
            out = h @ self._w(prefix + f"self_attn.{name}.weight").T
            return out.reshape(batch, length, n, cfg.head_dim).transpose(0, 2, 1, 3)

        q = self._rope(heads("q_proj", cfg.n_heads), positions)
        k = self._rope(heads("k_proj", cfg.n_kv_heads), positions)
        v = heads("v_proj", cfg.n_kv_heads)
        if past is not None:
            k = np.concatenate([past[0], k], axis=2)
            v = np.concatenate([past[1], v], axis=2)
        group = cfg.n_heads // cfg.n_kv_heads
        scores = q @ np.repeat(k, group, axis=1).transpose(0, 1, 3, 2)
        scores = scores * np.float32(cfg.attention_scale)
        # Row t of this call is token n_past + t and may see columns 0 .. n_past + t.
        allowed = np.arange(n_past + length)[None, :] <= (n_past + np.arange(length))[:, None]
        scores = np.where(allowed, scores, np.float32(-np.inf))
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        mixed = (probs @ np.repeat(v, group, axis=1)).transpose(0, 2, 1, 3)
        mixed = mixed.reshape(batch, length, cfg.d_model)
        x = x + (mixed @ self._w(prefix + "self_attn.o_proj.weight").T) * r

        h = _rms_norm(x, self._w(prefix + "post_attention_layernorm.weight"), cfg.norm_eps)
        if cfg.fused_mlp:
            both = h @ self._w(prefix + "shared_mlp.input_linear.weight").T
            gate, up = both[..., : cfg.d_ff], both[..., cfg.d_ff :]
            down = self._w(prefix + "shared_mlp.output_linear.weight")
        else:
            gate = h @ self._w(prefix + "mlp.gate_proj.weight").T
            up = h @ self._w(prefix + "mlp.up_proj.weight").T
            down = self._w(prefix + "mlp.down_proj.weight")
        x = x + ((_silu(gate) * up) @ down.T) * r
        return x.astype(np.float32, copy=False), (k, v)

    def head(self, x: np.ndarray) -> np.ndarray:
        """Step 8: final norm and output head, ``(..., d_model)`` to ``(..., vocab)``."""
        cfg = self.config
        x = _rms_norm(x, self._w("model.norm.weight"), cfg.norm_eps)
        name = "model.embed_tokens.weight" if cfg.tied_embeddings else "lm_head.weight"
        return (x @ self._w(name).T) / np.float32(cfg.logits_scaling)

    def hidden_states(self, tokens: Any) -> list[np.ndarray]:
        """The hidden state after the embedding and after every block, as numpy:
        ``n_layers + 1`` arrays of shape ``(T, d_model)`` (or ``(B, T, d_model)``).
        Matches ``output_hidden_states`` of ``transformers``."""
        ids, batched = self._ids(tokens)
        x = self.embed(ids)
        states = [x]
        for layer in range(self.config.n_layers):
            x, _ = self.block(x, layer)
            states.append(x)
        return states if batched else [state[0] for state in states]

    # -- the TinyLM-shaped call -----------------------------------------------------

    def _ids(self, tokens: Any) -> tuple[np.ndarray, bool]:
        ids = np.asarray(tokens.tolist() if hasattr(tokens, "tolist") else tokens)
        if ids.dtype.kind not in "iu" or ids.ndim not in (1, 2) or ids.size == 0:
            raise ValueError("tokens must be a non-empty 1-D or 2-D sequence of integer ids")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.config.vocab_size})")
        return (ids, True) if ids.ndim == 2 else (ids[None, :], False)

    def __call__(self, tokens: Any, cache: RealCache | None = None) -> tuple[Any, RealCache]:
        ids, batched = self._ids(tokens)
        if cache is not None:
            if not isinstance(cache, RealCache):
                raise TypeError("cache must be a RealCache returned by this model")
            if cache.keys[0].shape[0] != ids.shape[0]:
                raise ValueError("the cache and the tokens have different batch sizes")
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", int(ids.size))
        x = self.embed(ids)
        keys, values = [], []
        for layer in range(self.config.n_layers):
            past = None if cache is None else (cache.keys[layer], cache.values[layer])
            x, (k, v) = self.block(x, layer, past=past)
            k.setflags(write=False)
            v.setflags(write=False)
            keys.append(k)
            values.append(v)
        logits = self.head(x).astype(np.float32, copy=False)
        if not batched:
            logits = logits[0]
        length = (0 if cache is None else cache.length) + ids.shape[1]
        new_cache = RealCache(tuple(keys), tuple(values), length, batched)
        if self.fw is not None and getattr(self.fw, "name", "python") != "python":
            logits = self.fw.tensor(logits)
        return logits, new_cache

    def __repr__(self) -> str:
        cfg = self.config
        return f"RealLM({self.model.path.name!r}, layers={cfg.n_layers}, d_model={cfg.d_model})"
