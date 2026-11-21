"""A hybrid decoder: recurrent layers and attention layers in one stack.

The Qwen3.5 text decoders are mostly *not* attention: three of every four layers are
Gated DeltaNet layers (``learnai.kit.gdn``), which carry a fixed-size state, and every
fourth is an attention layer, which carries keys and values. This module hands a
solution the weights of a small model of that shape and expects it to run the forward
pass, including the decode loop over two kinds of per-layer state.

    cfg = HybridConfig()                       # 4 layers: linear, linear, linear, full
    params = hybrid_params(cfg, seed=0)        # dict of float32 numpy arrays
    logits, state = hybrid_forward_ref(cfg, params, prompt)
    logits, state = hybrid_forward_ref(cfg, params, [next_token], state)

The layout, block by block (pre-norm, as ``DecoderConfig``'s):

    x = x + mixer(rms_norm(x, input_layernorm))
    x = x + swiglu(rms_norm(x, post_attention_layernorm))

* Layer ``i`` is ``"full_attention"`` when ``(i + 1) % full_attention_interval == 0``
  and ``"linear_attention"`` otherwise; an explicit ``layer_types`` wins.
* A linear-attention mixer is ``ops.gated_delta_layer`` with the parameters under
  ``model.layers.{i}.linear_attn.``; its state is ``(conv_state, S)``.
* A full-attention mixer is *gated* attention. ``q_proj`` gives ``2 * head_dim``
  numbers per head: the first ``head_dim`` are the query, the last ``head_dim`` a
  gate. Queries and keys go through an RMS norm over the head dimension (``q_norm``,
  ``k_norm``) and then ``ops.partial_rope`` (the first ``int(head_dim *
  partial_rotary_factor)`` channels rotate); grouped-query attention with scores
  divided by ``sqrt(head_dim)``; then ``o_proj(attended * sigmoid(gate))``. Its state
  is ``(k, v)``: the normalized, rotated keys and the values of every token so far,
  ``(n_kv_heads, S, head_dim)`` each.
* Final RMSNorm, and a head tied to the embedding matrix.

Parameter names follow the checkpoint (``model.language_model.`` shortened to
``model.``). For a linear layer see ``learnai.kit.gdn``; for an attention layer:

    model.layers.{i}.self_attn.q_proj.weight   (n_heads * 2 * head_dim, d_model)
    model.layers.{i}.self_attn.k_proj.weight   (n_kv_heads * head_dim, d_model)
    model.layers.{i}.self_attn.v_proj.weight   (n_kv_heads * head_dim, d_model)
    model.layers.{i}.self_attn.o_proj.weight   (d_model, n_heads * head_dim)
    model.layers.{i}.self_attn.q_norm.weight   (head_dim,)
    model.layers.{i}.self_attn.k_norm.weight   (head_dim,)

and for every layer ``input_layernorm.weight``, ``post_attention_layernorm.weight``,
``mlp.gate_proj.weight``, ``mlp.up_proj.weight``, ``mlp.down_proj.weight``; then
``model.norm.weight`` and ``model.embed_tokens.weight``. ``head_dim`` is its own
setting, as in Qwen3.5: ``n_heads * head_dim`` need not equal ``d_model``.

One difference from the real checkpoints, on purpose: every norm weight here is a
plain gain (``rms_norm(x) * weight``). Qwen3.5 stores its block norms, final norm, and
QK norms zero-centered (``rms_norm(x) * (1 + weight)``); that convention is taught on
the real model, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from learnai.kit import ops
from learnai.kit.decoder import layer_params
from learnai.kit.gdn import GDNConfig, _layer

LINEAR, FULL = "linear_attention", "full_attention"


@dataclass(frozen=True)
class HybridConfig:
    """The shape of a hybrid decoder. Derived: ``layers``, ``rotary_dim``, ``gdn``."""

    vocab_size: int = 64
    d_model: int = 32
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 16
    d_mlp: int = 48
    rope_base: float = 10000.0
    partial_rotary_factor: float = 0.25
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    full_attention_interval: int = 4
    #: One of "linear_attention" / "full_attention" per layer; None applies the interval.
    layer_types: tuple[str, ...] | None = None
    n_key_heads: int = 2
    n_value_heads: int = 4
    key_head_dim: int = 8
    value_head_dim: int = 8
    conv_kernel: int = 4

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim",
                     "d_mlp", "full_attention_interval"):  # fmt: skip
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if self.rotary_dim % 2:
            raise ValueError(
                "int(head_dim * partial_rotary_factor) must be even: RoPE rotates pairs"
            )
        if self.layer_types is not None:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
            if len(self.layer_types) != self.n_layers:
                raise ValueError("layer_types needs one entry per layer")
            if set(self.layer_types) - {LINEAR, FULL}:
                raise ValueError(f"layer_types entries must be {LINEAR!r} or {FULL!r}")
        self.gdn  # noqa: B018 - validates the delta-layer sizes

    @property
    def layers(self) -> tuple[str, ...]:
        """The mixer of each layer: ``layer_types``, or the interval rule."""
        if self.layer_types is not None:
            return self.layer_types
        return tuple(
            FULL if (i + 1) % self.full_attention_interval == 0 else LINEAR
            for i in range(self.n_layers)
        )

    @property
    def rotary_dim(self) -> int:
        """How many channels of a head rotate: ``int(head_dim * partial_rotary_factor)``."""
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def gdn(self) -> GDNConfig:
        """The sizes of the linear-attention layers as a ``GDNConfig``."""
        return GDNConfig(
            d_model=self.d_model,
            n_key_heads=self.n_key_heads,
            n_value_heads=self.n_value_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            conv_kernel=self.conv_kernel,
            norm_eps=self.norm_eps,
        )


def hybrid_params(cfg: HybridConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded float32 weights for ``cfg``, keyed by checkpoint-style names.

    Equal ``(cfg, seed)`` give equal weights. Every gain is drawn from ``[0.5, 1.5]``
    and the delta-layer constants away from their initial values (``gdn_params``), so
    leaving any of them out changes the logits. With tied embeddings (the default)
    there is no ``lm_head.weight`` key. Logits have a standard deviation of about 2.
    """
    rng = np.random.default_rng([seed, 0x687962])
    d, attn, kv = cfg.d_model, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim

    def linear(out_dim: int, in_dim: int) -> np.ndarray:
        return (rng.standard_normal((out_dim, in_dim)) * in_dim**-0.5).astype(np.float32)

    def gain(size: int = d) -> np.ndarray:
        return rng.uniform(0.5, 1.5, size=size).astype(np.float32)

    params: dict[str, np.ndarray] = {}
    params["model.embed_tokens.weight"] = linear(cfg.vocab_size, d) * np.float32(2.0)
    for i, kind in enumerate(cfg.layers):
        layer = f"model.layers.{i}."
        params[layer + "input_layernorm.weight"] = gain()
        if kind == LINEAR:
            for name, value in _layer(cfg, rng).items():
                params[layer + "linear_attn." + name] = value
        else:
            params[layer + "self_attn.q_proj.weight"] = linear(2 * attn, d)
            params[layer + "self_attn.k_proj.weight"] = linear(kv, d)
            params[layer + "self_attn.v_proj.weight"] = linear(kv, d)
            params[layer + "self_attn.o_proj.weight"] = linear(d, attn)
            params[layer + "self_attn.q_norm.weight"] = gain(cfg.head_dim)
            params[layer + "self_attn.k_norm.weight"] = gain(cfg.head_dim)
        params[layer + "post_attention_layernorm.weight"] = gain()
        params[layer + "mlp.gate_proj.weight"] = linear(cfg.d_mlp, d)
        params[layer + "mlp.up_proj.weight"] = linear(cfg.d_mlp, d)
        params[layer + "mlp.down_proj.weight"] = linear(d, cfg.d_mlp)
    params["model.norm.weight"] = gain()
    if not cfg.tie_embeddings:
        params["lm_head.weight"] = linear(cfg.vocab_size, d) * np.float32(2.0)
    return params


def gated_attention_ref(
    cfg: HybridConfig,
    attn: dict[str, np.ndarray],
    h: np.ndarray,
    past: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """The full-attention mixer of the oracle: ``(out, (k, v))``.

    ``attn`` holds the ``self_attn.`` parameters with that prefix removed; ``h`` is the
    normed block input, ``(T, d_model)`` or ``(B, T, d_model)``; ``past`` is the
    ``(k, v)`` of the tokens before it. The new tokens take the positions after the
    cached ones and attend to everything cached and, causally, to one another.
    """
    lead, new = h.shape[:-2], h.shape[-2]
    seen = 0 if past is None else past[0].shape[-2]
    positions = np.arange(seen, seen + new)

    def heads(t: np.ndarray, n: int, width: int) -> np.ndarray:  # (.., T, n * w) -> (.., T, n, w)
        return t.reshape(*lead, new, n, width)

    # Each head's slice of q_proj is [query | gate]: split inside the head, not across.
    both = heads(h @ attn["q_proj.weight"].T, cfg.n_heads, 2 * cfg.head_dim)
    q, gate = both[..., : cfg.head_dim], both[..., cfg.head_dim :]
    k = heads(h @ attn["k_proj.weight"].T, cfg.n_kv_heads, cfg.head_dim)
    v = heads(h @ attn["v_proj.weight"].T, cfg.n_kv_heads, cfg.head_dim)
    q = ops.rms_norm(q, attn["q_norm.weight"], cfg.norm_eps)
    k = ops.rms_norm(k, attn["k_norm.weight"], cfg.norm_eps)
    q, k, v = (np.swapaxes(t, -2, -3) for t in (q, k, v))  # (.., heads, T, head_dim)
    q = ops.partial_rope(q, positions, cfg.rope_base, cfg.partial_rotary_factor)
    k = ops.partial_rope(k, positions, cfg.rope_base, cfg.partial_rotary_factor)
    if past is not None:
        k = np.concatenate((past[0], k), axis=-2)
        v = np.concatenate((past[1], v), axis=-2)
    # Query t sits at position seen + t and may attend to keys 0 .. seen + t.
    mask = np.arange(seen + new)[None, :] <= positions[:, None]
    attended = np.swapaxes(ops.attention(q, k, v, mask), -2, -3)  # (.., T, n_heads, head_dim)
    width = cfg.n_heads * cfg.head_dim
    gated = attended.reshape(*lead, new, width) * _sigmoid(gate.reshape(*lead, new, width))
    return gated @ attn["o_proj.weight"].T, (k.astype(np.float32), v.astype(np.float32))


def hybrid_forward_ref(
    cfg: HybridConfig,
    params: dict[str, np.ndarray],
    tokens: Any,
    state: list[Any] | None = None,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """The oracle forward pass, in numpy: ``(logits, state)``.

    ``tokens`` is an integer array (or list) of shape ``(T,)`` or ``(B, T)``, ``T >=
    1``; the logits are float32 ``(T, vocab_size)`` or ``(B, T, vocab_size)``, one row
    per token of this call. ``state`` is ``None`` (the tokens start a sequence) or the
    list a previous call returned (the tokens continue it): one entry per layer,
    ``(conv_state, S)`` for a linear-attention layer and ``(k, v)`` for a
    full-attention layer. The state passed in is not modified.

    Feeding a sequence in pieces agrees with feeding it at once to float32 rounding
    (about ``1e-5`` on the logits), not bit for bit.
    """
    ids = np.asarray(tokens)
    if ids.ndim not in (1, 2) or ids.size == 0 or ids.dtype.kind not in "iu":
        raise ValueError("tokens must be integers of shape (T,) or (B, T) with T >= 1")
    if ids.min() < 0 or ids.max() >= cfg.vocab_size:
        raise ValueError(f"token ids must be in [0, {cfg.vocab_size})")
    if state is None:
        state = [None] * cfg.n_layers
    if len(state) != cfg.n_layers:
        raise ValueError(f"state must have one entry per layer ({cfg.n_layers}), or be None")
    x = params["model.embed_tokens.weight"][ids]
    new_state: list[tuple[np.ndarray, np.ndarray]] = []
    for i, kind in enumerate(cfg.layers):
        x, kept = _block(cfg, kind, layer_params(params, i), x, state[i])
        new_state.append(kept)
    x = ops.rms_norm(x, params["model.norm.weight"], cfg.norm_eps)
    head = params["model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"]
    return (x @ head.T).astype(np.float32), new_state


def hybrid_block_ref(
    cfg: HybridConfig,
    params: dict[str, np.ndarray],
    layer: int,
    x: Any,
    state: Any = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """One block of the oracle, in numpy: ``(x, state)`` for layer ``layer``.

    ``x`` is the block's input, ``(T, d_model)`` or ``(B, T, d_model)``; ``state`` is
    that layer's entry of the forward pass's state (``(conv_state, S)`` or ``(k, v)``),
    or ``None`` when the tokens start a sequence. Chaining the blocks from the
    embedding and applying the final norm and the head gives ``hybrid_forward_ref``.
    """
    x = np.asarray(x, dtype=np.float32)
    return _block(cfg, cfg.layers[layer], layer_params(params, layer), x, state)


def _block(
    cfg: HybridConfig, kind: str, block: dict[str, np.ndarray], x: np.ndarray, state: Any
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """``x + mixer(norm(x))``, then ``x + swiglu(norm(x))``, for one layer's parameters
    (names without the layer prefix, every norm weight a plain gain)."""
    past = None if state is None else tuple(np.asarray(t, np.float32) for t in state)
    h = ops.rms_norm(x, block["input_layernorm.weight"], cfg.norm_eps)
    if kind == LINEAR:
        mixed, kept = ops.gated_delta_layer(
            h, _strip(block, "linear_attn."), past, cfg.n_key_heads, cfg.norm_eps
        )
    else:
        mixed, kept = gated_attention_ref(cfg, _strip(block, "self_attn."), h, past)
    x = x + mixed
    h = ops.rms_norm(x, block["post_attention_layernorm.weight"], cfg.norm_eps)
    x = x + ops.swiglu(
        h,
        block["mlp.gate_proj.weight"],
        block["mlp.up_proj.weight"],
        block["mlp.down_proj.weight"],
    )
    return x, kept


def _strip(block: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {name[len(prefix) :]: value for name, value in block.items() if name.startswith(prefix)}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))
