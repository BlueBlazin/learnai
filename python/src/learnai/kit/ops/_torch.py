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
