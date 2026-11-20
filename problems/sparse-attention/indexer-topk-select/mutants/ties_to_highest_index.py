# BUG: equal scores are ranked by index with the highest first instead of the lowest.
import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5)
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    masked = torch.where(visible, scores, float("-inf"))
    flipped = torch.flip(masked, dims=(-1,))
    order = torch.argsort(flipped, dim=-1, descending=True, stable=True)
    rank = torch.flip(torch.argsort(order, dim=-1), dims=(-1,))
    return (rank < k) & visible
