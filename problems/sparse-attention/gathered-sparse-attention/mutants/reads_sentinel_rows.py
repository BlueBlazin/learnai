# BUG: unused slots are clamped to row 0, read, and masked afterward: the output is right, but one row is read for every unused slot.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    valid = idx >= 0
    k_rows, v_rows = store.gather(idx.clamp(min=0))
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid[:, None, :], weights, 0.0)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
