"""The kit's building blocks in MLX. The contract is in ``learnai.kit.ops``.

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


def softplus(x: mx.array) -> mx.array:
    # log(1 + e^x) as max(x, 0) + log1p(e^-|x|): no overflow for large x, and a small
    # result (large negative x) is not rounded away the way log(1 + tiny) would be.
    # (mlx.nn.softplus is off by about 6e-4, relatively, at x = -10; this is not.)
    return mx.maximum(x, 0) + mx.log1p(mx.exp(-mx.abs(x)))


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


def causal_conv1d(
    x: mx.array, w: mx.array, state: mx.array | None = None
) -> tuple[mx.array, mx.array]:
    if w.ndim == 3:
        w = w[:, 0]  # the checkpoint layout (C, 1, K) of a depthwise Conv1d
    kernel, steps = w.shape[-1], x.shape[-2]
    if state is None:
        # Nothing came before: the K - 1 inputs to the left of the first one are zeros.
        state = mx.zeros((*x.shape[:-2], kernel - 1, x.shape[-1]), dtype=x.dtype)
    padded = mx.concatenate((state, x), axis=-2)  # (..., K - 1 + T, C)
    # Output t reads inputs t - K + 1 .. t: tap j of the kernel meets input t - K + 1 + j,
    # so the last tap multiplies the current input (cross-correlation, no flip).
    y = sum(padded[..., j : j + steps, :] * w[:, j] for j in range(kernel))
    return y, padded[..., padded.shape[-2] - (kernel - 1) :, :]


def partial_rope(
    x: mx.array, positions: mx.array, base: float = 10000.0, factor: float = 1.0
) -> mx.array:
    r = int(x.shape[-1] * factor)  # rotated channels: the first r
    if r % 2:
        raise ValueError(f"int(D * factor) must be even to rotate pairs, got {r}")
    half = r // 2
    # The rotated block is a complete RoPE of width r, with its own frequency ladder.
    freqs = mx.power(base, -mx.arange(0, r, 2, dtype=mx.float32) / r)
    angles = positions.astype(mx.float32)[..., None] * freqs  # (T, r/2) or (B, T, r/2)
    if positions.ndim == 2:
        angles = angles[:, None]  # (B, 1, T, r/2): one set of positions per batch row
    cos, sin = mx.cos(angles), mx.sin(angles)
    # Half-split inside the block: pair i is channels (i, i + r/2). The rest pass through.
    a, b = x[..., :half], x[..., half:r]
    return mx.concatenate((a * cos - b * sin, a * sin + b * cos, x[..., r:]), axis=-1)


def gated_delta_layer(
    x: mx.array,
    params: dict[str, mx.array],
    state: tuple[mx.array, mx.array] | None = None,
    n_key_heads: int | None = None,
    norm_eps: float = 1e-6,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    lead, steps = x.shape[:-2], x.shape[-2]
    n_v = params["A_log"].shape[0]  # value heads
    n_k = n_v if n_key_heads is None else n_key_heads  # key heads
    value_dim = params["in_proj_z.weight"].shape[0]
    key_dim = (params["in_proj_qkv.weight"].shape[0] - value_dim) // 2
    d_k, d_v = key_dim // n_k, value_dim // n_v  # per head
    conv_state, memory = (None, None) if state is None else state
    if memory is None:
        memory = mx.zeros((*lead, n_v, d_v, d_k), dtype=mx.float32)

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
    q, k = mx.repeat(q, n_v // n_k, axis=-2), mx.repeat(k, n_v // n_k, axis=-2)
    # 3. Gates, one per value head and token: decay g in (0, 1), write strength beta.
    a = x @ params["in_proj_a.weight"].T + params["dt_bias"]
    g = mx.exp(-mx.exp(params["A_log"]) * softplus(a))
    beta = 1.0 / (1.0 + mx.exp(-(x @ params["in_proj_b.weight"].T)))
    # 4. The gated delta rule, one token at a time. memory is (..., n_v, d_v, d_k).
    outputs = []
    for t in range(steps):
        k_t = k[..., t, :, :]
        memory = memory * g[..., t, :, None, None]  # decay
        predicted = (memory @ k_t[..., None])[..., 0]  # what the memory holds under k_t
        delta = beta[..., t, :, None] * (v[..., t, :, :] - predicted)
        memory = memory + delta[..., :, None] * k_t[..., None, :]  # write the correction
        outputs.append((memory @ q[..., t, :, :, None])[..., 0])  # read after the write
    y = mx.stack(outputs, axis=-3)  # (..., T, n_v, d_v)
    # 5. Norm per value head, then the gate (norm first), then the output projection.
    z = (x @ params["in_proj_z.weight"].T).reshape(*lead, steps, n_v, d_v)
    y = rms_norm(y, params["norm.weight"], norm_eps) * _silu(z)
    out = y.reshape(*lead, steps, value_dim) @ params["out_proj.weight"].T
    return out, (conv_state, memory)


def eb_accept(logits: mx.array, entropy_bound: float) -> mx.array:
    logp = log_softmax(logits)
    p = mx.exp(logp)
    # Entropy in nats; a token of probability 0 contributes 0, not 0 * -inf.
    entropy = -mx.sum(mx.where(p > 0, p * mx.where(p > 0, logp, 0.0), 0.0), axis=-1)
    order = mx.argsort(entropy, axis=-1)  # stable: ascending, ties to the lowest index
    ranked = mx.take_along_axis(entropy, order, axis=-1)
    # The entropy of the positions ranked strictly before each one...
    before = mx.cumsum(ranked, axis=-1) - ranked
    # ...and back in position order.
    before = mx.take_along_axis(before, mx.argsort(order, axis=-1), axis=-1)
    return before <= entropy_bound


def _silu(x: mx.array) -> mx.array:
    return x / (1.0 + mx.exp(-x))  # silu(x) = x * sigmoid(x)
