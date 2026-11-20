# BUG: the scores are scaled by k^-0.5, the number of slots, instead of d^-0.5, the head width.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    valid = idx >= 0
    k_rows, v_rows = store.gather(idx, valid=valid)
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * idx.shape[-1] ** -0.5
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid[:, None, :], weights, 0.0)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
