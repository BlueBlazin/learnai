# BUG: checks k = 0 but not k >= V, then reads the k-th largest value of each row; for k > V that column does not exist and the call raises instead of returning the logits unchanged.
import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True, stable=True)
    rank = torch.argsort(order, dim=-1)
    kth = sorted_logits[:, k - 1]  # the smallest kept value
    return torch.where((rank < k) & (logits >= kth[:, None]), logits, float("-inf"))
