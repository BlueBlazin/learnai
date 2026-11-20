# BUG: the unused slots are masked after the softmax instead of before it: their weights are set to zero, but the weights of the used slots were normalized together with them and no longer sum to 1.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    valid = idx >= 0
    k_rows, v_rows = store.gather(idx, valid=valid)
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    weights = torch.softmax(scores, dim=-1) * valid[:, None, :]
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
