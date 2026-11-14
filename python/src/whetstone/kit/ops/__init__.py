"""ops: reference building blocks, with one signature in every framework.

    from whetstone.kit import ops

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
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = [
    "attention",
    "log_softmax",
    "min_p_filter",
    "rms_norm",
    "rope",
    "softmax",
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
