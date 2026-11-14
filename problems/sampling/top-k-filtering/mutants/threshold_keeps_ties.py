# BUG: keeps every logit >= the k-th largest value, so all entries tied with the k-th one survive and a row can keep more than k tokens.
import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True, stable=True)
    threshold = sorted_logits[:, k - 1 : k]
    return torch.where(logits >= threshold, logits, float("-inf"))
