"""A real hybrid model: the text decoder of ``Qwen/Qwen3.5-0.8B`` from its raw tensors.

    model = RealHybrid(ctx.asset("qwen3.5-0.8b"))

    model.config            # HybridConfig read from config.json (24 layers, width 1024)
    model.config_json       # the raw config.json dict (text settings under "text_config")
    model.checkpoint        # SafetensorsFile: all 488 tensors, memory-mapped
    model.weights()         # name -> float32 tensor, exactly as stored, decoded lazily
    model.lm()              # RealHybridLM: the reference forward pass

The checkpoint is a vision-language model. The text decoder is the 320 tensors under
``model.language_model.``; ``model.visual.*`` (153 tensors) and ``mtp.*`` (15) are not
used by a text-only forward pass. There is no ``lm_head`` tensor: the output head is
the embedding matrix.

The decoder has the layout of the kit's ``HybridConfig`` / ``hybrid_forward_ref``
(``kit/hybrid.py``), and ``RealHybridLM`` runs that code, block by block. The checkpoint differs from
the kit's synthetic weights in four ways, and ``kit_name`` / ``RealHybridLM.params``
are the whole translation:

1. **Names.** ``model.language_model.layers.0...`` here, ``model.layers.0...`` there.
2. **Norm weights are stored as offsets from one.** The two block norms of every
   layer, the final norm, and ``q_norm`` / ``k_norm`` of the attention layers compute
   ``rms_norm(x) * (1 + weight)``. The gated norm inside a delta layer
   (``linear_attn.norm.weight``) is a plain gain, used as stored.
3. **Storage.** Everything is bfloat16 except ``linear_attn.A_log`` and
   ``linear_attn.norm.weight``, which are float32; all of it is used as float32.
4. **Sizes.** 24 layers by ``layer_types`` (attention at 3, 7, 11, 15, 19, 23), 16 key
   and 16 value heads of 128 in a delta layer, 8 query heads over 2 KV heads of 256 in
   an attention layer, of which the first 64 channels rotate (theta 1e7).

Only the standard library and numpy are needed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from learnai.kit import ops
from learnai.kit.counters import Counters
from learnai.kit.hybrid import HybridConfig, _block
from learnai.kit.realmodel import QWEN35_WEIGHTS_FILE, RealModel, Weights

#: Every text-decoder tensor name starts with this.
TEXT_PREFIX = "model.language_model."
#: Suffixes of the norm weights that are stored as ``gain - 1`` (see the module docs).
OFFSET_NORMS = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    "language_model.norm.weight",
)


def hybrid_config_from_json(raw: Mapping[str, Any]) -> HybridConfig:
    """The ``HybridConfig`` of a Qwen3.5 ``config.json`` (the whole file, or its
    ``text_config``). ``layer_types`` is taken when present, else the interval rule."""
    text = raw.get("text_config", raw)
    if text.get("model_type") != "qwen3_5_text":
        raise ValueError(
            f"unsupported model_type {text.get('model_type')!r}: RealHybrid reads the "
            "`qwen3_5_text` configuration of a Qwen3.5 checkpoint"
        )
    if text.get("mlp_only_layers") or not text.get("attn_output_gate", True):
        raise ValueError("a Qwen3.5 config with mlp_only_layers or without the output gate")
    rope = text.get("rope_parameters") or {}
    kinds = text.get("layer_types")
    return HybridConfig(
        vocab_size=int(text["vocab_size"]),
        d_model=int(text["hidden_size"]),
        n_layers=int(text["num_hidden_layers"]),
        n_heads=int(text["num_attention_heads"]),
        n_kv_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        d_mlp=int(text["intermediate_size"]),
        rope_base=float(rope["rope_theta"]),
        partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
        norm_eps=float(text["rms_norm_eps"]),
        tie_embeddings=bool(text.get("tie_word_embeddings", True)),
        full_attention_interval=int(text.get("full_attention_interval", 4)),
        layer_types=tuple(kinds) if kinds else None,
        n_key_heads=int(text["linear_num_key_heads"]),
        n_value_heads=int(text["linear_num_value_heads"]),
        key_head_dim=int(text["linear_key_head_dim"]),
        value_head_dim=int(text["linear_value_head_dim"]),
        conv_kernel=int(text["linear_conv_kernel_dim"]),
    )


def kit_name(name: str) -> str | None:
    """The name ``hybrid_params`` uses for a checkpoint tensor, or None when the text
    decoder does not use it: ``model.language_model.norm.weight`` -> ``model.norm.weight``."""
    if not name.startswith(TEXT_PREFIX):
        return None
    return "model." + name[len(TEXT_PREFIX) :]


def is_offset_norm(name: str) -> bool:
    """Whether a checkpoint tensor is a norm weight stored as ``gain - 1``."""
    return name.startswith(TEXT_PREFIX) and name.endswith(OFFSET_NORMS)


def checkpoint_layout(params: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Kit hybrid weights (``hybrid_params``) as the real checkpoint lays them out:
    names under ``model.language_model.`` and the block norms, final norm, and QK norms
    stored as ``gain - 1``. For the synthetic cases of a real-model problem, which run
    without the asset: a solution written for the checkpoint gives, on
    ``checkpoint_layout(hybrid_params(cfg, seed))``, the logits of
    ``hybrid_forward_ref(cfg, hybrid_params(cfg, seed), tokens)`` to float32 rounding.
    """
    out = {}
    for name, value in params.items():
        if not name.startswith("model."):
            raise ValueError(f"{name!r}: only a tied hybrid model has the checkpoint's layout")
        real = TEXT_PREFIX + name[len("model.") :]
        out[real] = value - np.float32(1.0) if is_offset_norm(real) else value
    return out


class RealHybrid(RealModel):
    """The downloaded ``qwen3.5-0.8b`` directory. See the module docs.

    ``weights()`` hands out the tensors under their checkpoint names with their stored
    values (norm weights still as offsets): what a porting problem gives the learner.
    ``tokenizer`` is not supported (the kit's byte-level BPE was not checked against
    this model's tokenizer); real-model problems on this model take token ids.
    """

    weights_file = QWEN35_WEIGHTS_FILE

    @property
    def config(self) -> HybridConfig:  # type: ignore[override]
        return hybrid_config_from_json(self.config_json)

    @property
    def tokenizer(self) -> Any:
        raise NotImplementedError(
            "RealHybrid has no tokenizer: problems on this model use token ids from fixtures"
        )

    def weights(self, fw: Any = None) -> Weights:
        """Name -> tensor with the stored values, lazily. Bound to a tensor framework
        the 3 GB of float32 weights are held once, as that framework's tensors, and
        not a second time as numpy (``Weights(keep_numpy=False)``)."""
        return Weights(self.checkpoint, fw, keep_numpy=False)

    def text_names(self) -> list[str]:
        """The names of the text decoder's tensors, in file order (320 of the 488)."""
        return [name for name in self.checkpoint.names() if name.startswith(TEXT_PREFIX)]

    def lm(self, fw: Any = None) -> RealHybridLM:
        return RealHybridLM(self, fw)


class RealHybridLM:
    """The reference forward pass of the Qwen3.5 text decoder, called like
    ``hybrid_forward_ref``.

        lm = RealHybrid(ctx.asset("qwen3.5-0.8b")).lm()
        logits, state = lm(tokens)                # tokens: (T,) or (B, T) ids
        logits, state = lm(next_tokens, state)    # continue the sequence

    ``logits`` has one row per token of the call, float32, in the framework the model
    is bound to. ``state`` is the kit's hybrid state, as numpy: one entry per layer,
    ``(conv_state, S)`` for a linear-attention layer (the last 3 projected inputs,
    ``(3, 6144)``, and the memory ``(16, 128, 128)`` as ``(heads, value, key)``) and
    ``(k, v)`` for a full-attention layer (normalized, rotated keys and values,
    ``(2, seen, 256)``); a batch axis comes first when the tokens have one.
    ``lm.counters`` counts ``forward_calls`` and ``tokens_processed``. numpy inside.

    What the forward pass does, in order: the complete list a port has to get right.

    1.  Tokens are embedded with ``model.language_model.embed_tokens.weight``; nothing
        is multiplied in or added (no scale, no position embedding).
    2.  24 pre-norm blocks: ``x = x + mixer(norm1(x))``, ``x = x + mlp(norm2(x))``.
    3.  Every block norm and the final norm is ``x * rsqrt(mean(x^2) + 1e-6) *
        (1 + weight)``: the stored weight is an offset from one.
    4.  The mixer of a ``"linear_attention"`` layer is ``ops.gated_delta_layer`` (the
        ``gated-delta-layer`` problem) with 16 key and 16 value heads of 128 and a
        ``(6144, 1, 4)`` convolution. Its own norm weight (``linear_attn.norm.weight``,
        128 numbers shared by the heads) is a plain gain, used as stored.
    5.  The mixer of a ``"full_attention"`` layer is gated attention: ``q_proj`` has
        ``8 * 2 * 256`` rows, per head the query then a gate; ``k_proj`` and ``v_proj``
        give 2 heads of 256, each serving 4 consecutive query heads; queries and keys
        go through an RMS norm over the head with ``1 + q_norm.weight`` and
        ``1 + k_norm.weight``; then RoPE on the first 64 channels of each head (theta
        1e7, half-split inside those 64: channel ``i`` pairs with ``i + 32``, pair
        ``i`` turns by ``p * theta ** (-i / 32)``), the other 192 pass through; scores
        are divided by ``sqrt(256)``; causal softmax; ``o_proj(attended *
        sigmoid(gate))``.
    6.  The MLP is SwiGLU of width 3584: ``down(silu(gate(x)) * up(x))``.
    7.  ``model.language_model.norm`` (item 3), then the head, which is the embedding
        matrix itself: there is no ``lm_head`` tensor.
    8.  Weights are bfloat16 on disk (two kinds of small tensor are float32) and are
        used as float32.

    The pieces are exposed for oracles: ``embed``, ``block``, ``final_norm``, ``head``,
    ``hidden_states``.
    """

    def __init__(self, model: RealHybrid, fw: Any = None, counters: Counters | None = None) -> None:
        self.model = model
        self.config = model.config
        self.fw = fw
        self.counters = counters or Counters("forward_calls", "tokens_processed")
        self._weights = model.weights()
        self._blocks: dict[int, dict[str, np.ndarray]] = {}

    def with_framework(self, fw: Any) -> RealHybridLM:
        """The same model bound to another framework, with fresh counters."""
        return RealHybridLM(self.model, fw)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    # -- the translation to the kit's layout ---------------------------------------------

    def _gain(self, name: str) -> np.ndarray:
        """A checkpoint tensor as the kit's forward uses it: offsets become gains."""
        value = self._weights.numpy(name)
        return np.float32(1.0) + value if is_offset_norm(name) else value

    def params(self, layer: int) -> dict[str, np.ndarray]:
        """One layer's tensors in the kit's layout: the names of ``layer_params(
        hybrid_params(cfg), layer)`` (``"input_layernorm.weight"``,
        ``"linear_attn.conv1d.weight"``, ...), every norm weight a plain gain."""
        if layer not in self._blocks:
            prefix = f"{TEXT_PREFIX}layers.{layer}."
            self._blocks[layer] = {
                name[len(prefix) :]: self._gain(name)
                for name in self.model.checkpoint.names()
                if name.startswith(prefix)
            }
        return self._blocks[layer]

    # -- pieces, for fixtures and oracles ---------------------------------------------------

    def embed(self, tokens: Any) -> np.ndarray:
        """Step 1: ``(..., T)`` ids to ``(..., T, d_model)`` float32."""
        return self._weights.numpy(TEXT_PREFIX + "embed_tokens.weight")[np.asarray(tokens)]

    def block(self, x: Any, layer: int, state: Any = None) -> tuple[np.ndarray, tuple]:
        """Steps 2 to 6 for one layer: ``(T, d_model)`` or ``(B, T, d_model)`` in and
        out, as numpy. ``state`` is this layer's entry of the state (None: the tokens
        start a sequence); the new entry is returned and the old one is not modified.
        """
        x = np.asarray(x, dtype=np.float32)
        x, kept = _block(self.config, self.config.layers[layer], self.params(layer), x, state)
        return x.astype(np.float32, copy=False), kept

    def final_norm(self, x: Any) -> np.ndarray:
        """``model.language_model.norm``: what ``transformers`` applies to the last entry
        of its ``hidden_states`` (and to ``last_hidden_state``)."""
        x = np.asarray(x, dtype=np.float32)
        return ops.rms_norm(x, self._gain(TEXT_PREFIX + "norm.weight"), self.config.norm_eps)

    def head(self, x: Any) -> np.ndarray:
        """Step 7: final norm and the tied head, ``(..., d_model)`` to ``(..., vocab)``."""
        x = self.final_norm(x)
        table = self._weights.numpy(TEXT_PREFIX + "embed_tokens.weight")
        return (x @ table.T).astype(np.float32, copy=False)

    def hidden_states(self, tokens: Any) -> list[np.ndarray]:
        """The hidden state after the embedding and after every block, as numpy:
        ``n_layers + 1`` arrays of shape ``(T, d_model)`` (or ``(B, T, d_model)``).
        Entry ``i + 1`` is the output of layer ``i``; none has the final norm applied.
        ``output_hidden_states`` of ``transformers`` is the same list except for its
        last entry, which is ``final_norm`` of this one.
        """
        x = self.embed(self._ids(tokens))
        states = [x]
        for layer in range(self.config.n_layers):
            x, _ = self.block(x, layer)
            states.append(x)
        return states

    # -- the call ----------------------------------------------------------------------------

    def _ids(self, tokens: Any) -> np.ndarray:
        ids = np.asarray(tokens.tolist() if hasattr(tokens, "tolist") else tokens)
        if ids.dtype.kind not in "iu" or ids.ndim not in (1, 2) or ids.size == 0:
            raise ValueError("tokens must be a non-empty 1-D or 2-D sequence of integer ids")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError(f"token ids must be in [0, {self.config.vocab_size})")
        return ids

    def __call__(self, tokens: Any, state: list[Any] | None = None) -> tuple[Any, list[tuple]]:
        ids = self._ids(tokens)
        n_layers = self.config.n_layers
        if state is None:
            state = [None] * n_layers
        if len(state) != n_layers:
            raise ValueError(f"state must have one entry per layer ({n_layers}), or be None")
        self.counters.add("forward_calls")
        self.counters.add("tokens_processed", int(ids.size))
        x = self.embed(ids)
        new_state = []
        for layer in range(n_layers):
            x, kept = self.block(x, layer, state[layer])
            new_state.append(kept)
        logits = self.head(x)
        if self.fw is not None and getattr(self.fw, "name", "python") != "python":
            logits = self.fw.tensor(logits)
        return logits, new_state

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"RealHybridLM({self.model.path.name!r}, layers={cfg.n_layers}, d_model={cfg.d_model})"
        )
