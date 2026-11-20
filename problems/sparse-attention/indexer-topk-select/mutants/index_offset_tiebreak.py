# BUG: ties are broken by subtracting 1e-6 times the index before torch.topk: for equal scores near 1000 the offset is below float32 resolution and breaks no tie, and distinct scores closer than 1e-6 are reordered.
import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5)
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    width = scores.shape[-1]
    masked = torch.where(visible, scores, float("-inf"))
    masked = masked - 1e-6 * torch.arange(width)
    chosen = torch.topk(masked, min(k, width), dim=-1).indices
    mask = torch.zeros_like(visible).scatter(-1, chosen, True)
    return mask & visible
