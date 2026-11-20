# BUG: the invisible entries are overwritten with -inf in the caller's `scores` tensor instead of in a copy.
import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5)
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    scores.masked_fill_(~visible, float("-inf"))
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    return (rank < k) & visible
