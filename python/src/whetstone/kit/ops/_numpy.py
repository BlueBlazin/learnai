"""The kit's building blocks in numpy. The contract is in ``whetstone.kit.ops``."""

from __future__ import annotations

import math

import numpy as np

NEG_INF = float("-inf")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # Shift so the largest entry is 0: no overflow, and the denominator is at least 1.
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * weight


def rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    dim = x.shape[-1]
    # One frequency per pair of features: theta_i = base^(-2i / D).
    theta = base ** (-np.arange(0, dim, 2, dtype=np.float32) / dim)
    angles = positions.astype(np.float32)[..., None] * theta  # (T, D/2) or (B, T, D/2)
    if positions.ndim == 2:
        angles = angles[:, None]  # (B, 1, T, D/2): one set of positions per batch row
    cos, sin = np.cos(angles), np.sin(angles)
    # Pair i is (x[..., 2i], x[..., 2i + 1]); rotate it counterclockwise by its angle.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return np.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1).reshape(x.shape)


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    k, v = _expand_kv_heads(q, k, v)
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        # A boolean mask says where attention is allowed; a float mask is added.
        scores = np.where(mask, scores, NEG_INF) if mask.dtype == np.bool_ else scores + mask
    # Softmax over the keys, except that a query with no visible key gets all-zero
    # weights (and so a zero output) instead of NaN.
    top = scores.max(axis=-1, keepdims=True)
    e = np.exp(scores - np.where(top == NEG_INF, 0.0, top))
    total = e.sum(axis=-1, keepdims=True)
    weights = e / np.where(total > 0, total, 1.0)
    return (weights @ v).astype(v.dtype)


def _expand_kv_heads(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grouped-query attention: repeat each key/value head for its group of query heads."""
    if q.ndim < 3 or k.ndim < 3 or q.shape[-3] == k.shape[-3]:
        return k, v
    groups, rest = divmod(q.shape[-3], k.shape[-3])
    if rest:
        raise ValueError(f"{q.shape[-3]} query heads cannot share {k.shape[-3]} key/value heads")
    # Query head h uses key/value head h // groups.
    return np.repeat(k, groups, axis=-3), np.repeat(v, groups, axis=-3)


def swiglu(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    gate = x @ w_gate.T
    with np.errstate(over="ignore"):  # exp(-z) may overflow to inf, and z / inf is 0
        silu = gate / (1.0 + np.exp(-gate))  # silu(z) = z * sigmoid(z)
    return (silu * (x @ w_up.T)) @ w_down.T


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    return np.where(_rank(-logits) < k, logits, NEG_INF).astype(logits.dtype)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    probs = softmax(logits)
    order = np.argsort(-probs, axis=-1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=-1)
    # Probability mass of the tokens ranked strictly before each one, in rank order...
    mass_before = np.cumsum(sorted_probs, axis=-1) - sorted_probs
    # ...and back in token order: the rank of a token is where `order` lists it.
    mass_before = np.take_along_axis(mass_before, np.argsort(order, axis=-1), axis=-1)
    return np.where(mass_before < p, logits, NEG_INF).astype(logits.dtype)


def min_p_filter(logits: np.ndarray, min_p: float) -> np.ndarray:
    probs = softmax(logits)
    keep = probs >= min_p * probs.max(axis=-1, keepdims=True)
    return np.where(keep, logits, NEG_INF).astype(logits.dtype)


def _rank(keys: np.ndarray) -> np.ndarray:
    """Rank of each entry along the last axis, ascending; ties go to the lowest index."""
    order = np.argsort(keys, axis=-1, kind="stable")
    return np.argsort(order, axis=-1)  # the inverse permutation
