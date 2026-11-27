import torch


def fill_masked(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    # where selects: the entry that is not kept is never used in any arithmetic, so it
    # does not matter what it holds.
    return torch.where(keep, scores, float("-inf"))


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # (B, T) -> (B, T, 1), so the mask lines up with the steps and broadcasts over the
    # features. A dropped step becomes 0, which adds nothing to the sum.
    kept = torch.where(mask[:, :, None], x, 0.0)
    count = mask.sum(dim=1)[:, None]  # (B, 1): kept steps per row
    # A row that keeps nothing has sum 0 and count 0; dividing by 1 returns its zeros.
    return kept.sum(dim=1) / count.clamp(min=1)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # A dropped step becomes -inf, which no kept value loses to. A row that keeps
    # nothing is all -inf, and so is its max.
    return torch.where(mask[:, :, None], x, float("-inf")).amax(dim=1)
