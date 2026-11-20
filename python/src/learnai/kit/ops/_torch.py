"""The kit's building blocks in PyTorch. The contract is in ``learnai.kit.ops``.

Everything is written from elementary operations: no ``torch.softmax``, ``topk``, or
``scaled_dot_product_attention``, so these work in problems that forbid the built-ins.
"""

from __future__ import annotations

import math

import torch

NEG_INF = float("-inf")


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    # Shift so the largest entry is 0: no overflow, and the denominator is at least 1.
    e = torch.exp(x - torch.amax(x, dim=axis, keepdim=True))
    return e / e.sum(dim=axis, keepdim=True)


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    shifted = x - torch.amax(x, dim=axis, keepdim=True)
    return shifted - torch.log(torch.exp(shifted).sum(dim=axis, keepdim=True))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps) * weight


def rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    dim = x.shape[-1]
    # One frequency per pair of features: theta_i = base^(-2i / D).
    theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim)
    angles = positions.to(torch.float32)[..., None] * theta  # (T, D/2) or (B, T, D/2)
    if positions.ndim == 2:
        angles = angles[:, None]  # (B, 1, T, D/2): one set of positions per batch row
    cos, sin = torch.cos(angles), torch.sin(angles)
    # Pair i is (x[..., 2i], x[..., 2i + 1]); rotate it counterclockwise by its angle.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).reshape(x.shape)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    k, v = _expand_kv_heads(q, k, v)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        # A boolean mask says where attention is allowed; a float mask is added.
        scores = torch.where(mask, scores, NEG_INF) if mask.dtype == torch.bool else scores + mask
    # Softmax over the keys, except that a query with no visible key gets all-zero
    # weights (and so a zero output) instead of NaN.
    top = torch.amax(scores, dim=-1, keepdim=True)
    e = torch.exp(scores - torch.where(top == NEG_INF, 0.0, top))
    total = e.sum(dim=-1, keepdim=True)
    weights = e / torch.where(total > 0, total, 1.0)
    return weights @ v


def _expand_kv_heads(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped-query attention: repeat each key/value head for its group of query heads."""
    if q.ndim < 3 or k.ndim < 3 or q.shape[-3] == k.shape[-3]:
        return k, v
    groups, rest = divmod(q.shape[-3], k.shape[-3])
    if rest:
        raise ValueError(f"{q.shape[-3]} query heads cannot share {k.shape[-3]} key/value heads")
    # Query head h uses key/value head h // groups.
    return k.repeat_interleave(groups, dim=-3), v.repeat_interleave(groups, dim=-3)


def swiglu(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    gate = x @ w_gate.T
    silu = gate / (1.0 + torch.exp(-gate))  # silu(z) = z * sigmoid(z)
    return (silu * (x @ w_up.T)) @ w_down.T


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    return torch.where(_rank(-logits) < k, logits, NEG_INF)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = softmax(logits)
    order = torch.argsort(-probs, dim=-1, stable=True)
    sorted_probs = torch.gather(probs, -1, order)
    # Probability mass of the tokens ranked strictly before each one, in rank order...
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    # ...and back in token order: the rank of a token is where `order` lists it.
    mass_before = torch.gather(mass_before, -1, torch.argsort(order, dim=-1))
    return torch.where(mass_before < p, logits, NEG_INF)


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = softmax(logits)
    keep = probs >= min_p * torch.amax(probs, dim=-1, keepdim=True)
    return torch.where(keep, logits, NEG_INF)


def _rank(keys: torch.Tensor) -> torch.Tensor:
    """Rank of each entry along the last axis, ascending; ties go to the lowest index."""
    order = torch.argsort(keys, dim=-1, stable=True)
    return torch.argsort(order, dim=-1)  # the inverse permutation


def softplus(x: torch.Tensor) -> torch.Tensor:
    # log(1 + e^x) as max(x, 0) + log1p(e^-|x|): no overflow for large x, and a small
    # result (large negative x) is not rounded away the way log(1 + tiny) would be.
    return torch.clamp(x, min=0) + torch.log1p(torch.exp(-torch.abs(x)))


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def causal_conv1d(
    x: torch.Tensor, w: torch.Tensor, state: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if w.ndim == 3:
        w = w[:, 0]  # the checkpoint layout (C, 1, K) of a depthwise Conv1d
    kernel, steps = w.shape[-1], x.shape[-2]
    if state is None:
        # Nothing came before: the K - 1 inputs to the left of the first one are zeros.
        state = torch.zeros(
            (*x.shape[:-2], kernel - 1, x.shape[-1]), dtype=x.dtype, device=x.device
        )
    padded = torch.cat((state, x), dim=-2)  # (..., K - 1 + T, C)
    # Output t reads inputs t - K + 1 .. t: tap j of the kernel meets input t - K + 1 + j,
    # so the last tap multiplies the current input (cross-correlation, no flip).
    y = sum(padded[..., j : j + steps, :] * w[:, j] for j in range(kernel))
    return y, padded[..., padded.shape[-2] - (kernel - 1) :, :]


def partial_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0, factor: float = 1.0
) -> torch.Tensor:
    r = int(x.shape[-1] * factor)  # rotated channels: the first r
    if r % 2:
        raise ValueError(f"int(D * factor) must be even to rotate pairs, got {r}")
    half = r // 2
    # The rotated block is a complete RoPE of width r, with its own frequency ladder.
    freqs = base ** (-torch.arange(0, r, 2, dtype=torch.float32, device=x.device) / r)
    angles = positions.to(torch.float32)[..., None] * freqs  # (T, r/2) or (B, T, r/2)
    if positions.ndim == 2:
        angles = angles[:, None]  # (B, 1, T, r/2): one set of positions per batch row
    cos, sin = torch.cos(angles), torch.sin(angles)
    # Half-split inside the block: pair i is channels (i, i + r/2). The rest pass through.
    a, b = x[..., :half], x[..., half:r]
    return torch.cat((a * cos - b * sin, a * sin + b * cos, x[..., r:]), dim=-1)


def gated_delta_layer(
    x: torch.Tensor,
    params: dict[str, torch.Tensor],
    state: tuple[torch.Tensor, torch.Tensor] | None = None,
    n_key_heads: int | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    lead, steps = x.shape[:-2], x.shape[-2]
    n_v = params["A_log"].shape[0]  # value heads
    n_k = n_v if n_key_heads is None else n_key_heads  # key heads
    value_dim = params["in_proj_z.weight"].shape[0]
    key_dim = (params["in_proj_qkv.weight"].shape[0] - value_dim) // 2
    d_k, d_v = key_dim // n_k, value_dim // n_v  # per head
    conv_state, memory = (None, None) if state is None else state
    if memory is None:
        memory = torch.zeros((*lead, n_v, d_v, d_k), dtype=torch.float32, device=x.device)

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
    q = torch.repeat_interleave(q, n_v // n_k, dim=-2)
    k = torch.repeat_interleave(k, n_v // n_k, dim=-2)
    # 3. Gates, one per value head and token: decay g in (0, 1), write strength beta.
    a = x @ params["in_proj_a.weight"].T + params["dt_bias"]
    g = torch.exp(-torch.exp(params["A_log"]) * softplus(a))
    beta = 1.0 / (1.0 + torch.exp(-(x @ params["in_proj_b.weight"].T)))
    # 4. The gated delta rule, one token at a time. memory is (..., n_v, d_v, d_k).
    outputs = []
    for t in range(steps):
        k_t = k[..., t, :, :]
        memory = memory * g[..., t, :, None, None]  # decay
        predicted = (memory @ k_t[..., None])[..., 0]  # what the memory holds under k_t
        delta = beta[..., t, :, None] * (v[..., t, :, :] - predicted)
        memory = memory + delta[..., :, None] * k_t[..., None, :]  # write the correction
        outputs.append((memory @ q[..., t, :, :, None])[..., 0])  # read after the write
    y = torch.stack(outputs, dim=-3)  # (..., T, n_v, d_v)
    # 5. Norm per value head, then the gate (norm first), then the output projection.
    z = (x @ params["in_proj_z.weight"].T).reshape(*lead, steps, n_v, d_v)
    y = rms_norm(y, params["norm.weight"], norm_eps) * _silu(z)
    out = y.reshape(*lead, steps, value_dim) @ params["out_proj.weight"].T
    return out, (conv_state, memory)


def eb_accept(logits: torch.Tensor, entropy_bound: float) -> torch.Tensor:
    logp = log_softmax(logits)
    p = torch.exp(logp)
    # Entropy in nats; a token of probability 0 contributes 0, not 0 * -inf.
    zero = torch.zeros_like(p)
    entropy = -torch.where(p > 0, p * torch.where(p > 0, logp, zero), zero).sum(dim=-1)
    order = torch.argsort(entropy, dim=-1, stable=True)  # ascending, ties: lowest index
    ranked = torch.gather(entropy, -1, order)
    # The entropy of the positions ranked strictly before each one...
    before = torch.cumsum(ranked, dim=-1) - ranked
    # ...and back in position order.
    before = torch.gather(before, -1, torch.argsort(order, dim=-1, stable=True))
    return before <= entropy_bound


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.exp(-x))  # silu(x) = x * sigmoid(x)
