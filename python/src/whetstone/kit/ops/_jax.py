"""The kit's building blocks in JAX. The contract is in ``whetstone.kit.ops``.

Everything is written from elementary operations: no ``jax.nn.softmax`` or
``jax.lax.top_k``, so these work in problems that forbid the built-ins. All of it can
be traced by ``jax.jit`` (``k`` in ``top_k_filter`` must be a Python int).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

NEG_INF = float("-inf")


def softmax(x: jax.Array, axis: int = -1) -> jax.Array:
    # Shift so the largest entry is 0: no overflow, and the denominator is at least 1.
    e = jnp.exp(x - jnp.max(x, axis=axis, keepdims=True))
    return e / jnp.sum(e, axis=axis, keepdims=True)


def log_softmax(x: jax.Array, axis: int = -1) -> jax.Array:
    shifted = x - jnp.max(x, axis=axis, keepdims=True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True))


def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-5) -> jax.Array:
    return x / jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def rope(x: jax.Array, positions: jax.Array, base: float = 10000.0) -> jax.Array:
    dim = x.shape[-1]
    # One frequency per pair of features: theta_i = base^(-2i / D).
    theta = base ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    angles = positions.astype(jnp.float32)[..., None] * theta  # (T, D/2) or (B, T, D/2)
    if positions.ndim == 2:
        angles = angles[:, None]  # (B, 1, T, D/2): one set of positions per batch row
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    # Pair i is (x[..., 2i], x[..., 2i + 1]); rotate it counterclockwise by its angle.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1).reshape(x.shape)


def attention(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    k, v = _expand_kv_heads(q, k, v)
    scores = q @ jnp.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        # A boolean mask says where attention is allowed; a float mask is added.
        scores = jnp.where(mask, scores, NEG_INF) if mask.dtype == jnp.bool_ else scores + mask
    # Softmax over the keys, except that a query with no visible key gets all-zero
    # weights (and so a zero output) instead of NaN.
    top = jnp.max(scores, axis=-1, keepdims=True)
    e = jnp.exp(scores - jnp.where(top == NEG_INF, 0.0, top))
    total = jnp.sum(e, axis=-1, keepdims=True)
    weights = e / jnp.where(total > 0, total, 1.0)
    return weights @ v


def _expand_kv_heads(q: jax.Array, k: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Grouped-query attention: repeat each key/value head for its group of query heads."""
    if q.ndim < 3 or k.ndim < 3 or q.shape[-3] == k.shape[-3]:
        return k, v
    groups, rest = divmod(q.shape[-3], k.shape[-3])
    if rest:
        raise ValueError(f"{q.shape[-3]} query heads cannot share {k.shape[-3]} key/value heads")
    # Query head h uses key/value head h // groups.
    return jnp.repeat(k, groups, axis=-3), jnp.repeat(v, groups, axis=-3)


def swiglu(x: jax.Array, w_gate: jax.Array, w_up: jax.Array, w_down: jax.Array) -> jax.Array:
    gate = x @ w_gate.T
    silu = gate / (1.0 + jnp.exp(-gate))  # silu(z) = z * sigmoid(z)
    return (silu * (x @ w_up.T)) @ w_down.T


def top_k_filter(logits: jax.Array, k: int) -> jax.Array:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    return jnp.where(_rank(-logits) < k, logits, NEG_INF)


def top_p_filter(logits: jax.Array, p: float) -> jax.Array:
    probs = softmax(logits)
    order = jnp.argsort(-probs, axis=-1)  # stable
    sorted_probs = jnp.take_along_axis(probs, order, axis=-1)
    # Probability mass of the tokens ranked strictly before each one, in rank order...
    mass_before = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs
    # ...and back in token order: the rank of a token is where `order` lists it.
    mass_before = jnp.take_along_axis(mass_before, jnp.argsort(order, axis=-1), axis=-1)
    return jnp.where(mass_before < p, logits, NEG_INF)


def min_p_filter(logits: jax.Array, min_p: float) -> jax.Array:
    probs = softmax(logits)
    keep = probs >= min_p * jnp.max(probs, axis=-1, keepdims=True)
    return jnp.where(keep, logits, NEG_INF)


def _rank(keys: jax.Array) -> jax.Array:
    """Rank of each entry along the last axis, ascending; ties go to the lowest index."""
    order = jnp.argsort(keys, axis=-1)  # stable: equal keys stay in index order
    return jnp.argsort(order, axis=-1)  # the inverse permutation
