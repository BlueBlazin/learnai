# BUG: every visible entry whose score is at least the k-th highest is selected, so equal scores at the cut are all kept and a row can select more than k entries.
import torch


def index_scores(q_idx: torch.Tensor, k_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    heads, dim = q_idx.shape[1], q_idx.shape[2]
    per_head = torch.relu(torch.einsum("thd,sd->ths", q_idx, k_idx) * dim**-0.5)
    return torch.einsum("th,ths->ts", w * heads**-0.5, per_head)


def topk_mask(scores: torch.Tensor, visible: torch.Tensor, k: int) -> torch.Tensor:
    masked = torch.where(visible, scores, float("-inf"))
    kth = torch.sort(masked, dim=-1, descending=True).values[:, min(k, scores.shape[-1]) - 1]
    return (masked >= kth[:, None]) & visible
