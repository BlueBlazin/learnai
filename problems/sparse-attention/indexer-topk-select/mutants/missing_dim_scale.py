# BUG: the dot products are not multiplied by D^-0.5, so every score is too large by the factor sqrt(D).
import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads = q_idx.shape[1]
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx))
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    masked = torch.where(visible, scores, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)  # rank[t, s]: the place of entry s in row t
    return (rank < k) & visible
