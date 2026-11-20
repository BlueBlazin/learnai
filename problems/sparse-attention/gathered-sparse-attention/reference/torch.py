from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    valid = idx >= 0  # (T, k): which slots hold a token
    # Read the selected rows only; an unused slot is not read and comes back as zeros.
    k_rows, v_rows = store.gather(idx, valid=valid)  # (T, k, H, d) each
    scores = torch.einsum("thd,tjhd->thj", q, k_rows) * q.shape[-1] ** -0.5
    # A zero row would score 0 and take weight: unused slots must score -inf.
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    # A row with no token at all is NaN after the softmax; it attends to nothing.
    weights = torch.where(valid[:, None, :], weights, 0.0)
    return torch.einsum("thj,tjhd->thd", weights, v_rows)
