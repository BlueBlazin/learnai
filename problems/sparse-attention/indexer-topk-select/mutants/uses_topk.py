# BUG: the selection is torch.topk, which does not promise an order among equal scores, so which of several equal scores at the cut is selected is arbitrary instead of the lowest index.
import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5)
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    masked = torch.where(visible, scores, float("-inf"))
    chosen = torch.topk(masked, min(k, scores.shape[-1]), dim=-1).indices
    mask = torch.zeros_like(visible).scatter(-1, chosen, True)
    return mask & visible
