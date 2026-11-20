# BUG: the ReLU is applied after the weighted sum over heads instead of to each head's dot product, so a head with a negative dot product subtracts from the score.
import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    dots = torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5
    return torch.relu(torch.einsum("th,ths->ts", w * heads**-0.5, dots))


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    masked = torch.where(visible, scores, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)  # rank[t, s]: the place of entry s in row t
    return (rank < k) & visible
