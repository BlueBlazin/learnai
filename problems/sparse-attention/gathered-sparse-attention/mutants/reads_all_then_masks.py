# BUG: every key and value of the store is read and the unselected ones are masked afterward: the output is right, but the rows read grow with the store instead of with the selection.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(store.n)
    keys, values = store.gather(rows)  # (S, H, d): the whole store
    selected = torch.zeros(q.shape[0], store.n + 1, dtype=torch.bool)
    selected = selected.scatter(1, torch.where(idx >= 0, idx, store.n), True)[:, :-1]
    scores = torch.einsum("thd,shd->ths", q, keys) * q.shape[-1] ** -0.5
    scores = torch.where(selected[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(selected[:, None, :], weights, 0.0)
    return torch.einsum("ths,shd->thd", weights, values)
