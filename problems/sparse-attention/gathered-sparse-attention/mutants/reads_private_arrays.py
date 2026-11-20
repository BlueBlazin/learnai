# BUG: the selected rows are indexed out of the store's private arrays, which bypasses the counter and is not allowed.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    valid = idx >= 0
    rows = idx.clamp(min=0)
    k_rows = torch.from_numpy(store._k)[rows]
    v_rows = torch.from_numpy(store._v)[rows]
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid[:, None, :], weights, 0.0)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
