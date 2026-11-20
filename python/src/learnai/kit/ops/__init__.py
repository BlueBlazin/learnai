"""ops: reference building blocks, with one signature in every framework.

    from learnai.kit import ops

    probs = ops.softmax(logits)                 # torch in, torch out; jax in, jax out; ...
    y = ops.attention(q, k, v, mask)

Each function is implemented natively in torch, JAX, MLX, and numpy (see ``_torch.py``,
``_jax.py``, ``_mlx.py``, ``_numpy.py`` next to this file: they are translations of one
another), and the call is routed by the type of its first argument. They let a problem
build on a concept from an earlier problem without asking for it again.

They are written from elementary operations (``exp``, ``sum``, ``sort``), never from a
framework's own ``softmax``, ``topk``, or attention kernel, so they keep working inside
problems that forbid those built-ins. Float32 in, float32 out.

The contract, identical in every framework:

``softmax(x, axis=-1)``, ``log_softmax(x, axis=-1)``
    Numerically stable; ``-inf`` entries get probability 0. A slice that is all
    ``-inf`` gives NaN.

``rms_norm(x, weight, eps=1e-5)``
    ``x / sqrt(mean(x**2, axis=-1) + eps) * weight`` over the last axis.

``rope(x, positions, base=10000.0)``
    Rotary position embedding of ``x`` with shape ``(..., T, D)``, ``D`` even. Feature
    pair ``i`` is ``(x[..., 2i], x[..., 2i + 1])``, rotated counterclockwise by
    ``positions[t] * base**(-2i / D)``. ``positions`` is an integer or float tensor of
    shape ``(T,)``, shared by all leading axes, or ``(B, T)`` for ``x`` of shape
    ``(B, H, T, D)``: one row of positions per batch row.

``attention(q, k, v, mask=None)``
    Scaled dot-product attention: ``softmax(q @ k^T / sqrt(d) + mask) @ v`` with ``q``
    of shape ``(..., T, d)``, ``k`` ``(..., S, d)``, ``v`` ``(..., S, d_v)``; the
    result is ``(..., T, d_v)``. ``mask`` broadcasts to ``(..., T, S)`` and is either
    boolean (True where the query may attend) or float (added to the scores: 0 to
    allow, ``-inf`` to block). A query that may attend to nothing gets a zero output,
    not NaN. Grouped-query attention is built in: with ``q`` of shape
    ``(B, H, T, d)`` and ``k``, ``v`` of shape ``(B, H_kv, S, d)``, ``H`` a multiple of
    ``H_kv``, query head ``h`` uses key/value head ``h // (H // H_kv)``.

``swiglu(x, w_gate, w_up, w_down)``
    ``(silu(x @ w_gate^T) * (x @ w_up^T)) @ w_down^T``; weights are ``(out, in)``.

``top_k_filter(logits, k)``, ``top_p_filter(logits, p)``, ``min_p_filter(logits, min_p)``
    Sampler filters over the last axis. Each returns the logits with every removed
    entry set to ``-inf`` and every kept entry unchanged, so they chain, and
    ``softmax`` of the result is the renormalized distribution.

    * ``top_k_filter`` keeps the ``k`` largest logits; equal logits are ranked by
      index, lowest first. ``k <= 0`` or ``k >= V`` changes nothing.
    * ``top_p_filter`` ranks tokens by decreasing probability (ties: lowest index
      first) and keeps a token while the probability mass ranked strictly before it is
      below ``p``. The top token always survives.
    * ``min_p_filter`` keeps tokens whose probability is at least ``min_p`` times the
      largest probability.

    Entries that are already ``-inf`` stay ``-inf``.

The pieces of the 2026 hybrid and diffusion models (Qwen3.5, DiffusionGemma), with the
conventions of their shipped implementations:

``softplus(x)``
    ``log(1 + exp(x))``, accurate for large negative ``x`` too (the result there is
    tiny, and a naive ``log(1 + tiny)`` rounds it away).

``l2norm(x, eps=1e-6)``
    ``x * rsqrt(sum(x**2, axis=-1) + eps)``: unit length over the last axis.

``causal_conv1d(x, w, state=None) -> (y, state)``
    Depthwise causal convolution of ``x`` ``(..., T, C)`` along ``T``. ``w`` is
    ``(C, K)`` or the checkpoint's ``(C, 1, K)``; no bias, no flip (cross-correlation):
    ``y[t, c] = sum_j w[c, j] * padded[t + j, c]`` where ``padded`` is ``state``
    followed by ``x``, so the last tap multiplies the current input. ``state`` holds
    the ``K - 1`` inputs before ``x`` (``(..., K - 1, C)``; ``None`` means zeros) and
    the returned state the last ``K - 1`` rows of ``padded``. Feeding a sequence in
    pieces, passing the state along, gives the output of one call.

``partial_rope(x, positions, base=10000.0, factor=1.0)``
    The default partial rotary embedding (Qwen3.5), as in the ``partial-rope``
    problem: with ``r = int(D * factor)``, the first ``r`` channels of ``x``
    ``(..., T, D)`` are a complete RoPE of width ``r`` in the half-split layout (pair
    ``i`` is channels ``(i, i + r/2)``, frequency ``base**(-2i / r)``) and the other
    ``D - r`` channels pass through. ``r`` must be even. ``positions`` as for
    ``rope``. Note the layout: ``rope`` pairs adjacent channels, so
    ``partial_rope(x, p, base, 1.0)`` is a permutation of ``rope(x, p, base)``, not
    equal to it.

``gated_delta_layer(x, params, state=None, n_key_heads=None, norm_eps=1e-6) -> (out, state)``
    The Gated DeltaNet token mixer (the ``linear_attn`` sublayer of Qwen3.5) on ``x``
    of shape ``(..., T, D)``. ``params`` is a dict with the checkpoint's names
    (``learnai.kit.gdn_params`` makes one; sizes are read from its shapes):
    ``in_proj_qkv.weight``, ``in_proj_z.weight``, ``in_proj_a.weight``,
    ``in_proj_b.weight``, ``conv1d.weight``, ``A_log``, ``dt_bias``, ``norm.weight``,
    ``out_proj.weight``. ``n_key_heads`` defaults to the number of value heads
    (``len(A_log)``); it cannot be read from the shapes. ``state`` is
    ``(conv_state, S)``: the ``K - 1`` last projected inputs ``(..., K - 1, C)`` and
    the memory ``(..., n_value_heads, d_value, d_key)``; ``None`` means zeros. In order:
    project; ``causal_conv1d``; SiLU; split into ``q``, ``k``, ``v``; ``l2norm`` of
    ``q`` and ``k`` and ``q * d_key**-0.5``; each key head repeated over its
    consecutive value heads; ``g = exp(-exp(A_log) * softplus(a + dt_bias))`` and
    ``beta = sigmoid(b)``; per token ``S = g S``, ``S = S + beta (v - S k) k^T``,
    ``y = S q``; ``norm.weight * rms_norm(y) * silu(z)`` per value head; ``out_proj``.
    One token at a time, so any chunking of a sequence agrees with one call to
    float32 rounding.

``eb_accept(logits, entropy_bound)``
    The entropy-bound acceptance of DiffusionGemma for logits ``(..., L, V)``: a
    boolean ``(..., L)`` mask. Positions are ranked by the entropy (in nats) of their
    distribution, ascending, ties to the lowest index; a position is accepted while
    the entropies ranked strictly before it sum to at most ``entropy_bound``. The
    lowest-entropy position is always accepted. Pass the logits the tokens were
    sampled from (after the temperature).
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = [
    "attention",
    "causal_conv1d",
    "eb_accept",
    "gated_delta_layer",
    "l2norm",
    "log_softmax",
    "min_p_filter",
    "partial_rope",
    "rms_norm",
    "rope",
    "softmax",
    "softplus",
    "swiglu",
    "top_k_filter",
    "top_p_filter",
]

_BACKENDS = {"torch": "_torch", "jax": "_jax", "jaxlib": "_jax", "mlx": "_mlx", "numpy": "_numpy"}


def _backend(tensor: Any) -> ModuleType:
    """The implementation module for the framework ``tensor`` belongs to."""
    package = (type(tensor).__module__ or "").split(".")[0]
    if package not in _BACKENDS:
        raise TypeError(
            f"ops functions take a torch, JAX, MLX, or numpy array as their first argument, "
            f"got {type(tensor).__name__}"
        )
    # Imported on first use, so importing the kit does not import every framework.
    return importlib.import_module(f"{__name__}.{_BACKENDS[package]}")


def softmax(x: Any, axis: int = -1) -> Any:
    """Softmax along ``axis``."""
    return _backend(x).softmax(x, axis)


def log_softmax(x: Any, axis: int = -1) -> Any:
    """Logarithm of the softmax along ``axis``, computed without leaving log space."""
    return _backend(x).log_softmax(x, axis)


def rms_norm(x: Any, weight: Any, eps: float = 1e-5) -> Any:
    """RMSNorm over the last axis, with gain ``weight``."""
    return _backend(x).rms_norm(x, weight, eps)


def rope(x: Any, positions: Any, base: float = 10000.0) -> Any:
    """Rotate ``x`` of shape ``(..., T, D)`` by its positions (rotary embedding)."""
    return _backend(x).rope(x, positions, base)


def attention(q: Any, k: Any, v: Any, mask: Any = None) -> Any:
    """Scaled dot-product attention, with grouped-query support and an optional mask."""
    return _backend(q).attention(q, k, v, mask)


def swiglu(x: Any, w_gate: Any, w_up: Any, w_down: Any) -> Any:
    """The SwiGLU MLP; weights are ``(out, in)``."""
    return _backend(x).swiglu(x, w_gate, w_up, w_down)


def top_k_filter(logits: Any, k: int) -> Any:
    """Keep the ``k`` largest logits of each row; the rest become ``-inf``."""
    return _backend(logits).top_k_filter(logits, k)


def top_p_filter(logits: Any, p: float) -> Any:
    """Keep the nucleus of probability mass ``p`` of each row; the rest become ``-inf``."""
    return _backend(logits).top_p_filter(logits, p)


def min_p_filter(logits: Any, min_p: float) -> Any:
    """Keep tokens at least ``min_p`` times as probable as the top one; the rest ``-inf``."""
    return _backend(logits).min_p_filter(logits, min_p)


def softplus(x: Any) -> Any:
    """``log(1 + exp(x))``, accurate for large negative ``x``."""
    return _backend(x).softplus(x)


def l2norm(x: Any, eps: float = 1e-6) -> Any:
    """``x * rsqrt(sum(x**2) + eps)`` over the last axis."""
    return _backend(x).l2norm(x, eps)


def causal_conv1d(x: Any, w: Any, state: Any = None) -> tuple[Any, Any]:
    """Depthwise causal convolution with its ``K - 1`` rows of state: ``(y, state)``."""
    return _backend(x).causal_conv1d(x, w, state)


def partial_rope(x: Any, positions: Any, base: float = 10000.0, factor: float = 1.0) -> Any:
    """Rotate the first ``int(D * factor)`` channels of ``x`` (half-split); pass the rest."""
    return _backend(x).partial_rope(x, positions, base, factor)


def gated_delta_layer(
    x: Any,
    params: dict[str, Any],
    state: Any = None,
    n_key_heads: int | None = None,
    norm_eps: float = 1e-6,
) -> tuple[Any, tuple[Any, Any]]:
    """The Gated DeltaNet token mixer with its state: ``(out, (conv_state, S))``."""
    return _backend(x).gated_delta_layer(x, params, state, n_key_heads, norm_eps)


def eb_accept(logits: Any, entropy_bound: float) -> Any:
    """Which positions the entropy-bound sampler accepts: a boolean mask."""
    return _backend(logits).eb_accept(logits, entropy_bound)
