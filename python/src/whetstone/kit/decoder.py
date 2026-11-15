"""A modern decoder layout: seeded parameters named like a checkpoint, and a numpy oracle.

``TinyLM`` is a black box that solutions call. This module is the opposite: it hands a
solution the *weights* of a small decoder-only transformer and expects the solution to
run the forward pass itself. The layout is the one current open models share (Llama,
Qwen, Granite):

* token embedding, no position embedding;
* ``n_layers`` pre-norm blocks: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``;
* RMSNorm; rotary position embedding (RoPE) on queries and keys;
* grouped-query attention: ``n_heads`` query heads share ``n_kv_heads`` key/value heads;
* a SwiGLU MLP; a final RMSNorm; an output head tied to the embedding matrix.

    cfg = DecoderConfig(n_layers=2)
    params = decoder_params(cfg, seed=0)              # dict of float32 numpy arrays
    logits = decoder_forward_ref(cfg, params, tokens)  # the oracle, numpy

Parameter names follow Hugging Face checkpoints of this family. Linear weights are
stored ``(out, in)`` and have no bias. For layer ``i``:

    model.embed_tokens.weight                         (vocab_size, d_model)
    model.layers.{i}.input_layernorm.weight           (d_model,)
    model.layers.{i}.self_attn.q_proj.weight          (n_heads * head_dim, d_model)
    model.layers.{i}.self_attn.k_proj.weight          (n_kv_heads * head_dim, d_model)
    model.layers.{i}.self_attn.v_proj.weight          (n_kv_heads * head_dim, d_model)
    model.layers.{i}.self_attn.o_proj.weight          (d_model, n_heads * head_dim)
    model.layers.{i}.post_attention_layernorm.weight  (d_model,)
    model.layers.{i}.mlp.gate_proj.weight             (d_mlp, d_model)
    model.layers.{i}.mlp.up_proj.weight               (d_mlp, d_model)
    model.layers.{i}.mlp.down_proj.weight             (d_model, d_mlp)
    model.norm.weight                                 (d_model,)
    lm_head.weight                                    (vocab_size, d_model), only if not tied

With ``tie_embeddings=True`` (the default) there is no ``lm_head.weight`` key: the
head is ``model.embed_tokens.weight``, as in a checkpoint with tied embeddings.

Every piece of arithmetic is one of the kit's ``ops`` (``rms_norm``, ``rope``,
``attention``, ``swiglu``), with their conventions: RoPE rotates adjacent feature pairs
with positions starting at 0, and attention scores are divided by ``sqrt(head_dim)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from whetstone.kit import ops


@dataclass(frozen=True)
class DecoderConfig:
    """The shape of a decoder. ``head_dim`` is derived: ``d_model // n_heads``."""

    vocab_size: int = 64
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    d_mlp: int = 48
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "n_kv_heads", "d_mlp"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim (d_model // n_heads) must be even for RoPE")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def decoder_params(cfg: DecoderConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded float32 weights for ``cfg``, keyed by checkpoint-style names.

    Equal ``(cfg, seed)`` give equal weights. Norm gains are drawn from ``[0.5, 1.5]``
    rather than set to 1, so leaving a norm out, or using the wrong one, changes the
    output. The dict is in checkpoint order: embedding, layers, final norm, head.
    """
    rng = np.random.default_rng(seed)
    d, kv = cfg.d_model, cfg.n_kv_heads * cfg.head_dim

    def linear(out_dim: int, in_dim: int) -> np.ndarray:
        return (rng.standard_normal((out_dim, in_dim)) * in_dim**-0.5).astype(np.float32)

    def gain() -> np.ndarray:
        return rng.uniform(0.5, 1.5, size=d).astype(np.float32)

    params: dict[str, np.ndarray] = {}
    # The embedding doubles as the output head, so its scale sets the spread of the
    # logits: a standard deviation of about 2, far above float32 rounding.
    params["model.embed_tokens.weight"] = linear(cfg.vocab_size, d) * np.float32(2.0)
    for i in range(cfg.n_layers):
        layer = f"model.layers.{i}."
        params[layer + "input_layernorm.weight"] = gain()
        params[layer + "self_attn.q_proj.weight"] = linear(d, d)
        params[layer + "self_attn.k_proj.weight"] = linear(kv, d)
        params[layer + "self_attn.v_proj.weight"] = linear(kv, d)
        params[layer + "self_attn.o_proj.weight"] = linear(d, d)
        params[layer + "post_attention_layernorm.weight"] = gain()
        params[layer + "mlp.gate_proj.weight"] = linear(cfg.d_mlp, d)
        params[layer + "mlp.up_proj.weight"] = linear(cfg.d_mlp, d)
        params[layer + "mlp.down_proj.weight"] = linear(d, cfg.d_mlp)
    params["model.norm.weight"] = gain()
    if not cfg.tie_embeddings:
        params["lm_head.weight"] = linear(cfg.vocab_size, d) * np.float32(2.0)
    return params


def layer_params(params: dict[str, Any], layer: int) -> dict[str, Any]:
    """The parameters of one block, with the ``model.layers.{layer}.`` prefix removed.

        block = layer_params(params, 0)
        block["self_attn.q_proj.weight"]

    Works on numpy arrays and on tensors alike (it only renames keys). Raises
    ``KeyError`` if the layer does not exist.
    """
    prefix = f"model.layers.{layer}."
    found = {
        name[len(prefix) :]: value for name, value in params.items() if name.startswith(prefix)
    }
    if not found:
        raise KeyError(f"no parameters for layer {layer}")
    return found


def decoder_forward_ref(
    cfg: DecoderConfig, params: dict[str, np.ndarray], tokens: Any
) -> np.ndarray:
    """The oracle forward pass, in numpy.

    ``tokens`` is an integer array (or list) of shape ``(T,)`` or ``(B, T)``; the
    result is float32 logits of shape ``(T, vocab_size)`` or ``(B, T, vocab_size)``.
    Every position attends causally and positions count from 0, so row ``t`` depends
    only on ``tokens[..., : t + 1]``.
    """
    ids = np.asarray(tokens)
    if ids.ndim not in (1, 2) or ids.size == 0 or ids.dtype.kind not in "iu":
        raise ValueError("tokens must be integers of shape (T,) or (B, T) with T >= 1")
    if ids.min() < 0 or ids.max() >= cfg.vocab_size:
        raise ValueError(f"token ids must be in [0, {cfg.vocab_size})")
    batched = ids.ndim == 2
    if not batched:
        ids = ids[None]
    seq = ids.shape[1]
    x = params["model.embed_tokens.weight"][ids]  # (B, T, D)
    causal = np.tril(np.ones((seq, seq), dtype=bool))  # query t may attend to keys <= t
    for i in range(cfg.n_layers):
        x = decoder_block_ref(cfg, layer_params(params, i), x, causal)
    x = ops.rms_norm(x, params["model.norm.weight"], cfg.norm_eps)
    head = params["model.embed_tokens.weight" if cfg.tie_embeddings else "lm_head.weight"]
    logits = (x @ head.T).astype(np.float32)
    return logits if batched else logits[0]


def decoder_block_ref(
    cfg: DecoderConfig,
    block: dict[str, np.ndarray],
    x: np.ndarray,
    mask: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """One pre-norm block of the oracle: ``x`` of shape ``(B, T, d_model)`` in and out.

    ``block`` is ``layer_params(params, i)``. ``mask`` follows ``ops.attention``
    (boolean, True where attention is allowed, or additive float) and must broadcast
    to ``(B, n_heads, T, T)``; ``None`` means no mask at all, not a causal one.
    ``positions`` follows ``ops.rope`` and defaults to ``0 .. T - 1``.
    """
    batch, seq, _ = x.shape
    if positions is None:
        positions = np.arange(seq)

    def heads(t: np.ndarray, n: int) -> np.ndarray:  # (B, T, n * head_dim) -> (B, n, T, head_dim)
        return t.reshape(batch, seq, n, cfg.head_dim).transpose(0, 2, 1, 3)

    h = ops.rms_norm(x, block["input_layernorm.weight"], cfg.norm_eps)
    q = heads(h @ block["self_attn.q_proj.weight"].T, cfg.n_heads)
    k = heads(h @ block["self_attn.k_proj.weight"].T, cfg.n_kv_heads)
    v = heads(h @ block["self_attn.v_proj.weight"].T, cfg.n_kv_heads)
    # Position enters through the rotation of queries and keys only; values are not rotated.
    q, k = ops.rope(q, positions, cfg.rope_base), ops.rope(k, positions, cfg.rope_base)
    attended = ops.attention(q, k, v, mask)  # (B, n_heads, T, head_dim)
    attended = attended.transpose(0, 2, 1, 3).reshape(batch, seq, cfg.d_model)
    x = x + attended @ block["self_attn.o_proj.weight"].T

    h = ops.rms_norm(x, block["post_attention_layernorm.weight"], cfg.norm_eps)
    return x + ops.swiglu(
        h,
        block["mlp.gate_proj.weight"],
        block["mlp.up_proj.weight"],
        block["mlp.down_proj.weight"],
    )
