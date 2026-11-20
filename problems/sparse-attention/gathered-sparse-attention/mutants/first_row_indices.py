# BUG: the first query's selection is used for every query: one row of idx is read and broadcast over all T queries.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    first = idx[:1]
    valid = first >= 0
    k_rows, v_rows = store.gather(first, valid=valid)  # (1, k, H, d)
    scores = torch.einsum("thd,jhd->thj", q, k_rows[0]) * q.shape[-1] ** -0.5
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid[:, None, :], weights, 0.0)
    return torch.einsum("thj,jhd->thd", weights, v_rows[0])
