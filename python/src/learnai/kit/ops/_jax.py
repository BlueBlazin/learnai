"""The kit's building blocks in JAX. The contract is in ``learnai.kit.ops``.

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


def softplus(x: jax.Array) -> jax.Array:
    # log(1 + e^x) as max(x, 0) + log1p(e^-|x|): no overflow for large x, and a small
    # result (large negative x) is not rounded away the way log(1 + tiny) would be.
    return jnp.maximum(x, 0) + jnp.log1p(jnp.exp(-jnp.abs(x)))


def l2norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def causal_conv1d(
    x: jax.Array, w: jax.Array, state: jax.Array | None = None
) -> tuple[jax.Array, jax.Array]:
    if w.ndim == 3:
        w = w[:, 0]  # the checkpoint layout (C, 1, K) of a depthwise Conv1d
    kernel, steps = w.shape[-1], x.shape[-2]
    if state is None:
        # Nothing came before: the K - 1 inputs to the left of the first one are zeros.
        state = jnp.zeros((*x.shape[:-2], kernel - 1, x.shape[-1]), dtype=x.dtype)
    padded = jnp.concatenate((state, x), axis=-2)  # (..., K - 1 + T, C)
    # Output t reads inputs t - K + 1 .. t: tap j of the kernel meets input t - K + 1 + j,
    # so the last tap multiplies the current input (cross-correlation, no flip).
    y = sum(padded[..., j : j + steps, :] * w[:, j] for j in range(kernel))
    return y, padded[..., padded.shape[-2] - (kernel - 1) :, :]


def partial_rope(
    x: jax.Array, positions: jax.Array, base: float = 10000.0, factor: float = 1.0
) -> jax.Array:
    r = int(x.shape[-1] * factor)  # rotated channels: the first r
    if r % 2:
        raise ValueError(f"int(D * factor) must be even to rotate pairs, got {r}")
    half = r // 2
    # The rotated block is a complete RoPE of width r, with its own frequency ladder.
    freqs = base ** (-jnp.arange(0, r, 2, dtype=jnp.float32) / r)
    angles = positions.astype(jnp.float32)[..., None] * freqs  # (T, r/2) or (B, T, r/2)
    if positions.ndim == 2:
        angles = angles[:, None]  # (B, 1, T, r/2): one set of positions per batch row
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    # Half-split inside the block: pair i is channels (i, i + r/2). The rest pass through.
    a, b = x[..., :half], x[..., half:r]
    return jnp.concatenate((a * cos - b * sin, a * sin + b * cos, x[..., r:]), axis=-1)


def gated_delta_layer(
    x: jax.Array,
    params: dict[str, jax.Array],
    state: tuple[jax.Array, jax.Array] | None = None,
    n_key_heads: int | None = None,
    norm_eps: float = 1e-6,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    lead, steps = x.shape[:-2], x.shape[-2]
    n_v = params["A_log"].shape[0]  # value heads
    n_k = n_v if n_key_heads is None else n_key_heads  # key heads
    value_dim = params["in_proj_z.weight"].shape[0]
    key_dim = (params["in_proj_qkv.weight"].shape[0] - value_dim) // 2
    d_k, d_v = key_dim // n_k, value_dim // n_v  # per head
    conv_state, memory = (None, None) if state is None else state
    if memory is None:
        memory = jnp.zeros((*lead, n_v, d_v, d_k), dtype=jnp.float32)

    # 1. Project, mix each channel with its last K - 1 inputs, SiLU, split.
    mixed, conv_state = causal_conv1d(
        x @ params["in_proj_qkv.weight"].T, params["conv1d.weight"], conv_state
    )
    mixed = _silu(mixed)
    q = mixed[..., :key_dim].reshape(*lead, steps, n_k, d_k)
    k = mixed[..., key_dim : 2 * key_dim].reshape(*lead, steps, n_k, d_k)
    v = mixed[..., 2 * key_dim :].reshape(*lead, steps, n_v, d_v)
    # 2. Unit-length queries and keys; the query also carries the 1 / sqrt(d_k) scale.
    q, k = l2norm(q) * d_k**-0.5, l2norm(k)
    # Key head j serves the n_v / n_k consecutive value heads j * n_v / n_k, ...
    q, k = jnp.repeat(q, n_v // n_k, axis=-2), jnp.repeat(k, n_v // n_k, axis=-2)
    # 3. Gates, one per value head and token: decay g in (0, 1), write strength beta.
    a = x @ params["in_proj_a.weight"].T + params["dt_bias"]
    g = jnp.exp(-jnp.exp(params["A_log"]) * softplus(a))
    beta = 1.0 / (1.0 + jnp.exp(-(x @ params["in_proj_b.weight"].T)))
    # 4. The gated delta rule, one token at a time. memory is (..., n_v, d_v, d_k).
    outputs = []
    for t in range(steps):
        k_t = k[..., t, :, :]
        memory = memory * g[..., t, :, None, None]  # decay
        predicted = (memory @ k_t[..., None])[..., 0]  # what the memory holds under k_t
        delta = beta[..., t, :, None] * (v[..., t, :, :] - predicted)
        memory = memory + delta[..., :, None] * k_t[..., None, :]  # write the correction
        outputs.append((memory @ q[..., t, :, :, None])[..., 0])  # read after the write
    y = jnp.stack(outputs, axis=-3)  # (..., T, n_v, d_v)
    # 5. Norm per value head, then the gate (norm first), then the output projection.
    z = (x @ params["in_proj_z.weight"].T).reshape(*lead, steps, n_v, d_v)
    y = rms_norm(y, params["norm.weight"], norm_eps) * _silu(z)
    out = y.reshape(*lead, steps, value_dim) @ params["out_proj.weight"].T
    return out, (conv_state, memory)


def eb_accept(logits: jax.Array, entropy_bound: float) -> jax.Array:
    logp = log_softmax(logits)
    p = jnp.exp(logp)
    # Entropy in nats; a token of probability 0 contributes 0, not 0 * -inf.
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.where(p > 0, logp, 0.0), 0.0), axis=-1)
    order = jnp.argsort(entropy, axis=-1)  # stable: ascending, ties to the lowest index
    ranked = jnp.take_along_axis(entropy, order, axis=-1)
    # The entropy of the positions ranked strictly before each one...
    before = jnp.cumsum(ranked, axis=-1) - ranked
    # ...and back in position order.
    before = jnp.take_along_axis(before, jnp.argsort(order, axis=-1), axis=-1)
    return before <= entropy_bound


def _silu(x: jax.Array) -> jax.Array:
    return x / (1.0 + jnp.exp(-x))  # silu(x) = x * sigmoid(x)
