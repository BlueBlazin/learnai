"""The kit's building blocks in MLX. The contract is in ``whetstone.kit.ops``.

Everything is written from elementary operations: no ``mx.softmax``, ``mx.topk``, or
``mx.fast`` kernels, so these work in problems that forbid the built-ins.
"""

from __future__ import annotations

import math

import mlx.core as mx

NEG_INF = float("-inf")


def softmax(x: mx.array, axis: int = -1) -> mx.array:
    # Shift so the largest entry is 0: no overflow, and the denominator is at least 1.
    e = mx.exp(x - mx.max(x, axis=axis, keepdims=True))
    return e / mx.sum(e, axis=axis, keepdims=True)


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    shifted = x - mx.max(x, axis=axis, keepdims=True)
    return shifted - mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True))


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-5) -> mx.array:
    return x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def rope(x: mx.array, positions: mx.array, base: float = 10000.0) -> mx.array:
    dim = x.shape[-1]
    # One frequency per pair of features: theta_i = base^(-2i / D).
    theta = mx.power(base, -mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    angles = positions.astype(mx.float32)[..., None] * theta  # (T, D/2) or (B, T, D/2)
    if positions.ndim == 2:
        angles = angles[:, None]  # (B, 1, T, D/2): one set of positions per batch row
    cos, sin = mx.cos(angles), mx.sin(angles)
    # Pair i is (x[..., 2i], x[..., 2i + 1]); rotate it counterclockwise by its angle.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return mx.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1).reshape(x.shape)


def attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
    k, v = _expand_kv_heads(q, k, v)
    scores = q @ mx.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        # A boolean mask says where attention is allowed; a float mask is added.
        scores = mx.where(mask, scores, NEG_INF) if mask.dtype == mx.bool_ else scores + mask
    # Softmax over the keys, except that a query with no visible key gets all-zero
    # weights (and so a zero output) instead of NaN.
    top = mx.max(scores, axis=-1, keepdims=True)
    e = mx.exp(scores - mx.where(top == NEG_INF, 0.0, top))
    total = mx.sum(e, axis=-1, keepdims=True)
    weights = e / mx.where(total > 0, total, 1.0)
    return weights @ v


def _expand_kv_heads(q: mx.array, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
    """Grouped-query attention: repeat each key/value head for its group of query heads."""
    if q.ndim < 3 or k.ndim < 3 or q.shape[-3] == k.shape[-3]:
        return k, v
    groups, rest = divmod(q.shape[-3], k.shape[-3])
    if rest:
        raise ValueError(f"{q.shape[-3]} query heads cannot share {k.shape[-3]} key/value heads")
    # Query head h uses key/value head h // groups.
    return mx.repeat(k, groups, axis=-3), mx.repeat(v, groups, axis=-3)


def swiglu(x: mx.array, w_gate: mx.array, w_up: mx.array, w_down: mx.array) -> mx.array:
    gate = x @ w_gate.T
    silu = gate / (1.0 + mx.exp(-gate))  # silu(z) = z * sigmoid(z)
    return (silu * (x @ w_up.T)) @ w_down.T


def top_k_filter(logits: mx.array, k: int) -> mx.array:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    return mx.where(_rank(-logits) < k, logits, NEG_INF)


def top_p_filter(logits: mx.array, p: float) -> mx.array:
    probs = softmax(logits)
    order = mx.argsort(-probs, axis=-1)  # stable
    sorted_probs = mx.take_along_axis(probs, order, axis=-1)
    # Probability mass of the tokens ranked strictly before each one, in rank order...
    mass_before = mx.cumsum(sorted_probs, axis=-1) - sorted_probs
    # ...and back in token order: the rank of a token is where `order` lists it.
    mass_before = mx.take_along_axis(mass_before, mx.argsort(order, axis=-1), axis=-1)
    return mx.where(mass_before < p, logits, NEG_INF)


def min_p_filter(logits: mx.array, min_p: float) -> mx.array:
    probs = softmax(logits)
    keep = probs >= min_p * mx.max(probs, axis=-1, keepdims=True)
    return mx.where(keep, logits, NEG_INF)


def _rank(keys: mx.array) -> mx.array:
    """Rank of each entry along the last axis, ascending; ties go to the lowest index."""
    order = mx.argsort(keys, axis=-1)  # stable: equal keys stay in index order
    return mx.argsort(order, axis=-1)  # the inverse permutation
