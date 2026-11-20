# BUG: idx is used as it is: -1 indexes the last row of the store, so every unused slot reads the last token's key and value and attends to it.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    k_rows, v_rows = store.gather(idx)
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
