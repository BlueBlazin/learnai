# BUG: a query whose slots are all unused takes a softmax over scores that are all -inf and returns NaN instead of zeros.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    valid = idx >= 0
    k_rows, v_rows = store.gather(idx, valid=valid)
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
