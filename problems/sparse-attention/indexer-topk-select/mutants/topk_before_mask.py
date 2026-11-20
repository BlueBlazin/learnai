# BUG: the top-k is taken over all entries and intersected with `visible` afterward, so a row whose best scores are invisible keeps fewer than k visible entries.
import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5)
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    return (rank < k) & visible
