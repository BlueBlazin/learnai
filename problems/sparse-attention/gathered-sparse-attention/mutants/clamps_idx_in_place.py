# BUG: the -1 entries of the caller's idx tensor are overwritten with 0 in place; the output and the rows read are right, but the caller's selection is destroyed.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    valid = idx >= 0
    idx[~valid] = 0
    k_rows, v_rows = store.gather(idx, valid=valid)
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(valid[:, None, :], weights, 0.0)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
