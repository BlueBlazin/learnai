# BUG: unused slots are not read but keep the score 0 of their zero rows instead of -inf, so each takes weight exp(0) in the softmax and dilutes the output.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    valid = idx >= 0
    k_rows, v_rows = store.gather(idx, valid=valid)
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
