"""The kit's building blocks in numpy. The contract is in ``learnai.kit.ops``."""

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


def softplus(x: np.ndarray) -> np.ndarray:
    # log(1 + e^x) as max(x, 0) + log1p(e^-|x|): no overflow for large x, and a small
    # result (large negative x) is not rounded away the way log(1 + tiny) would be.
    return (np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))).astype(x.dtype)


def l2norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x / np.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)).astype(x.dtype)


def causal_conv1d(
    x: np.ndarray, w: np.ndarray, state: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if w.ndim == 3:
        w = w[:, 0]  # the checkpoint layout (C, 1, K) of a depthwise Conv1d
    kernel, steps = w.shape[-1], x.shape[-2]
    if state is None:
        # Nothing came before: the K - 1 inputs to the left of the first one are zeros.
        state = np.zeros((*x.shape[:-2], kernel - 1, x.shape[-1]), dtype=x.dtype)
    padded = np.concatenate((state, x), axis=-2)  # (..., K - 1 + T, C)
    # Output t reads inputs t - K + 1 .. t: tap j of the kernel meets input t - K + 1 + j,
    # so the last tap multiplies the current input (cross-correlation, no flip).
    y = sum(padded[..., j : j + steps, :] * w[:, j] for j in range(kernel))
    return y.astype(x.dtype), padded[..., padded.shape[-2] - (kernel - 1) :, :]


def partial_rope(
    x: np.ndarray, positions: np.ndarray, base: float = 10000.0, factor: float = 1.0
) -> np.ndarray:
    r = int(x.shape[-1] * factor)  # rotated channels: the first r
    if r % 2:
        raise ValueError(f"int(D * factor) must be even to rotate pairs, got {r}")
    half = r // 2
    # The rotated block is a complete RoPE of width r, with its own frequency ladder.
    freqs = base ** (-np.arange(0, r, 2, dtype=np.float32) / r)
    angles = positions.astype(np.float32)[..., None] * freqs  # (T, r/2) or (B, T, r/2)
    if positions.ndim == 2:
        angles = angles[:, None]  # (B, 1, T, r/2): one set of positions per batch row
    cos, sin = np.cos(angles), np.sin(angles)
    # Half-split inside the block: pair i is channels (i, i + r/2). The rest pass through.
    a, b = x[..., :half], x[..., half:r]
    return np.concatenate((a * cos - b * sin, a * sin + b * cos, x[..., r:]), axis=-1)


def gated_delta_layer(
    x: np.ndarray,
    params: dict[str, np.ndarray],
    state: tuple[np.ndarray, np.ndarray] | None = None,
    n_key_heads: int | None = None,
    norm_eps: float = 1e-6,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    lead, steps = x.shape[:-2], x.shape[-2]
    n_v = params["A_log"].shape[0]  # value heads
    n_k = n_v if n_key_heads is None else n_key_heads  # key heads
    value_dim = params["in_proj_z.weight"].shape[0]
    key_dim = (params["in_proj_qkv.weight"].shape[0] - value_dim) // 2
    d_k, d_v = key_dim // n_k, value_dim // n_v  # per head
    conv_state, memory = (None, None) if state is None else state
    if memory is None:
        memory = np.zeros((*lead, n_v, d_v, d_k), dtype=np.float32)

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
    q, k = np.repeat(q, n_v // n_k, axis=-2), np.repeat(k, n_v // n_k, axis=-2)
    # 3. Gates, one per value head and token: decay g in (0, 1), write strength beta.
    a = x @ params["in_proj_a.weight"].T + params["dt_bias"]
    g = np.exp(-np.exp(params["A_log"]) * softplus(a))
    beta = _sigmoid(x @ params["in_proj_b.weight"].T)
    # 4. The gated delta rule, one token at a time. memory is (..., n_v, d_v, d_k).
    outputs = []
    for t in range(steps):
        k_t = k[..., t, :, :]
        memory = memory * g[..., t, :, None, None]  # decay
        predicted = (memory @ k_t[..., None])[..., 0]  # what the memory holds under k_t
        delta = beta[..., t, :, None] * (v[..., t, :, :] - predicted)
        memory = memory + delta[..., :, None] * k_t[..., None, :]  # write the correction
        outputs.append((memory @ q[..., t, :, :, None])[..., 0])  # read after the write
    y = np.stack(outputs, axis=-3)  # (..., T, n_v, d_v)
    # 5. Norm per value head, then the gate (norm first), then the output projection.
    z = (x @ params["in_proj_z.weight"].T).reshape(*lead, steps, n_v, d_v)
    y = rms_norm(y, params["norm.weight"], norm_eps) * _silu(z)
    out = y.reshape(*lead, steps, value_dim) @ params["out_proj.weight"].T
    return out.astype(np.float32), (conv_state, memory.astype(np.float32))


def eb_accept(logits: np.ndarray, entropy_bound: float) -> np.ndarray:
    logp = log_softmax(logits)
    p = np.exp(logp)
    # Entropy in nats; a token of probability 0 contributes 0, not 0 * -inf.
    entropy = -np.where(p > 0, p * np.where(p > 0, logp, 0.0), 0.0).sum(axis=-1)  # (..., L)
    order = np.argsort(entropy, axis=-1, kind="stable")  # ascending, ties: lowest index
    ranked = np.take_along_axis(entropy, order, axis=-1)
    # The entropy of the positions ranked strictly before each one...
    before = np.cumsum(ranked, axis=-1) - ranked
    # ...and back in position order.
    before = np.take_along_axis(before, np.argsort(order, axis=-1), axis=-1)
    return before <= entropy_bound


def _silu(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):  # exp(-x) may overflow to inf, and x / inf is 0
        return x / (1.0 + np.exp(-x))  # silu(x) = x * sigmoid(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))
