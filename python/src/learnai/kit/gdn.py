"""Gated DeltaNet: seeded parameters named like a checkpoint, and numpy oracles.

A Gated DeltaNet layer is the recurrent token mixer of the Qwen3.5 hybrid models (its
``linear_attn`` sublayer). Where attention keeps one key and one value per token, this
layer keeps, per value head, one matrix ``S`` of shape ``(d_value, d_key)`` that maps
keys to values, and updates it with every token:

    S = g * S                              forget a little (g in (0, 1))
    S = S + beta * (v - S k) k^T           make S k closer to v (the delta rule)
    y = S q                                read with the query, after the write

    cfg = GDNConfig()                                  # d_model=32, 2 key heads, 4 value heads
    params = gdn_params(cfg, seed=0)                   # dict of float32 numpy arrays
    out, state = gdn_layer_ref(cfg, params, x)         # x (T, d_model) or (B, T, d_model)
    out2, state = gdn_layer_ref(cfg, params, more, state)

Parameter names are those of the ``linear_attn`` module of a Qwen3.5 checkpoint
(``model.language_model.layers.{i}.linear_attn.*``), linear weights ``(out, in)``, no
biases:

    in_proj_qkv.weight   (2 * key_dim + value_dim, d_model)   q, k, v, in that order
    in_proj_z.weight     (value_dim, d_model)                 the output gate
    in_proj_a.weight     (n_value_heads, d_model)             decay
    in_proj_b.weight     (n_value_heads, d_model)             write strength
    conv1d.weight        (2 * key_dim + value_dim, 1, conv_kernel)
    A_log, dt_bias       (n_value_heads,)
    norm.weight          (value_head_dim,)                    shared by all value heads
    out_proj.weight      (d_model, value_dim)

``key_dim`` and ``value_dim`` are totals over heads, as in the shipped configuration:
``key_dim = n_key_heads * key_head_dim``.

The layer itself is ``learnai.kit.ops.gated_delta_layer`` (one signature in every
framework); ``gdn_layer_ref`` is that function in numpy with the sizes of ``cfg``.
The state is ``(conv_state, S)``: the last ``conv_kernel - 1`` projected inputs,
``(conv_kernel - 1, 2 * key_dim + value_dim)``, and the memory, ``(n_value_heads,
value_head_dim, key_head_dim)``. Neither can be rebuilt from the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from learnai.kit import ops
from learnai.kit.ops import _numpy as _np_ops


@dataclass(frozen=True)
class GDNConfig:
    """The shape of a Gated DeltaNet layer. Qwen3.5-0.8B: 16 and 16 heads of 128, kernel 4."""

    d_model: int = 32
    n_key_heads: int = 2
    n_value_heads: int = 4
    key_head_dim: int = 8
    value_head_dim: int = 8
    conv_kernel: int = 4
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("d_model", "n_key_heads", "n_value_heads", "key_head_dim",
                     "value_head_dim", "conv_kernel"):  # fmt: skip
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.n_value_heads % self.n_key_heads:
            raise ValueError("n_value_heads must be a multiple of n_key_heads")

    @property
    def key_dim(self) -> int:
        """Width of all key heads together (the queries have the same width)."""
        return self.n_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        """Width of all value heads together."""
        return self.n_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        """Channels of the convolution: queries, keys, and values side by side."""
        return 2 * self.key_dim + self.value_dim


def gdn_params(cfg: Any, seed: int = 0) -> dict[str, np.ndarray]:
    """Seeded float32 weights of one layer, keyed by the checkpoint's names.

    ``cfg`` is a ``GDNConfig`` (or a ``HybridConfig``, which has the same fields).
    Equal ``(cfg, seed)`` give equal weights. Nothing is at its initialization value:
    the gain ``norm.weight`` is drawn from ``[0.5, 1.5]``, ``dt_bias`` from
    ``[-1, 1]``, and ``exp(A_log)`` from ``[0.05, 0.5]``, so that leaving any of them
    out changes the output. With these ranges the decay ``g`` of a typical token lies
    between about 0.6 and 0.99: the memory spans tens of tokens.
    """
    return _layer(cfg, np.random.default_rng([seed, 0x67646E]))


def _layer(cfg: Any, rng: np.random.Generator) -> dict[str, np.ndarray]:
    d, conv = cfg.d_model, 2 * cfg.n_key_heads * cfg.key_head_dim
    value_dim = cfg.n_value_heads * cfg.value_head_dim
    conv += value_dim

    def linear(out_dim: int, in_dim: int) -> np.ndarray:
        return (rng.standard_normal((out_dim, in_dim)) * in_dim**-0.5).astype(np.float32)

    return {
        "in_proj_qkv.weight": linear(conv, d),
        "in_proj_z.weight": linear(value_dim, d),
        "in_proj_a.weight": linear(cfg.n_value_heads, d),
        "in_proj_b.weight": linear(cfg.n_value_heads, d),
        "conv1d.weight": (
            rng.standard_normal((conv, 1, cfg.conv_kernel)) * cfg.conv_kernel**-0.5
        ).astype(np.float32),
        "A_log": np.log(rng.uniform(0.05, 0.5, size=cfg.n_value_heads)).astype(np.float32),
        "dt_bias": rng.uniform(-1.0, 1.0, size=cfg.n_value_heads).astype(np.float32),
        "norm.weight": rng.uniform(0.5, 1.5, size=cfg.value_head_dim).astype(np.float32),
        "out_proj.weight": linear(d, value_dim),
    }


def gdn_gates_ref(a: Any, b: Any, a_log: Any, dt_bias: Any) -> tuple[np.ndarray, np.ndarray]:
    """The two gates: ``g = exp(-exp(A_log) * softplus(a + dt_bias))``, ``beta = sigmoid(b)``.

    ``a`` and ``b`` are ``(..., n_heads)`` (the projections ``in_proj_a`` and
    ``in_proj_b`` of the layer input). float32 numpy. ``g`` is in ``(0, 1]``.
    """
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    rate = np.exp(np.asarray(a_log, dtype=np.float32))
    g = np.exp(-rate * _np_ops.softplus(a + np.asarray(dt_bias, dtype=np.float32)))
    return g.astype(np.float32), _np_ops._sigmoid(b).astype(np.float32)


def gdn_step_ref(
    memory: Any, q: Any, k: Any, v: Any, g: Any, beta: Any
) -> tuple[np.ndarray, np.ndarray]:
    """One token of the gated delta rule for ``H`` heads: ``(y, memory)``.

    ``memory`` is ``(H, d_value, d_key)``, ``q`` and ``k`` ``(H, d_key)``, ``v``
    ``(H, d_value)``, ``g`` and ``beta`` ``(H,)``. Decay, write, then read:

        S = g * S;   S = S + beta * (v - S k) k^T;   y = S q

    ``q`` and ``k`` are used as given (the layer normalizes them first). The state
    layout is the one of ``mlx-lm``; ``transformers`` stores its transpose.
    """
    memory, q, k, v, g, beta = (np.asarray(t, dtype=np.float32) for t in (memory, q, k, v, g, beta))
    memory = memory * g[:, None, None]
    predicted = (memory @ k[:, :, None])[:, :, 0]
    memory = memory + (beta[:, None] * (v - predicted))[:, :, None] * k[:, None, :]
    return (memory @ q[:, :, None])[:, :, 0].astype(np.float32), memory.astype(np.float32)


def gdn_layer_ref(
    cfg: Any,
    params: dict[str, np.ndarray],
    x: Any,
    state: tuple[Any, Any] | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """The layer oracle: ``ops.gated_delta_layer`` in numpy, with the sizes of ``cfg``.

    ``x`` is ``(T, d_model)`` or ``(B, T, d_model)`` with ``T >= 1``; ``state`` is
    ``None`` (a fresh sequence) or the ``(conv_state, S)`` an earlier call returned.
    Returns ``(out, (conv_state, S))``, float32.
    """
    rows = np.asarray(x, dtype=np.float32)
    if rows.ndim not in (2, 3) or rows.shape[-2] < 1 or rows.shape[-1] != cfg.d_model:
        raise ValueError(
            f"x must have shape (T, {cfg.d_model}) or (B, T, {cfg.d_model}) with T >= 1, "
            f"got {rows.shape}"
        )
    if state is not None:
        state = (np.asarray(state[0], dtype=np.float32), np.asarray(state[1], dtype=np.float32))
    weights = {name: np.asarray(value, dtype=np.float32) for name, value in params.items()}
    return ops.gated_delta_layer(rows, weights, state, cfg.n_key_heads, cfg.norm_eps)
