# BUG: the keys and values are taken from store.reference(), which is not counted and is not allowed, and masked full attention is computed from them.
from typing import Any

import torch


def sparse_attend(q: torch.Tensor, store: Any, idx: torch.Tensor) -> torch.Tensor:
    keys, values = (torch.from_numpy(a) for a in store.reference())
    selected = torch.zeros(q.shape[0], store.n + 1, dtype=torch.bool)
    selected = selected.scatter(1, torch.where(idx >= 0, idx, store.n), True)[:, :-1]
    scores = torch.einsum("thd,shd->ths", q, keys) * q.shape[-1] ** -0.5
    scores = torch.where(selected[:, None, :], scores, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(selected[:, None, :], weights, 0.0)
    return torch.einsum("ths,shd->thd", weights, values)
