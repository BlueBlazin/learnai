import torch


def token_scores(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    # gather along the last axis wants an index with as many axes as logits:
    # (B, T) -> (B, T, 1). The result has the index's shape, and the size-1 axis that
    # was added is then removed by name.
    return torch.gather(logits, 2, ids[:, :, None]).squeeze(2)


def take_columns(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # The index already has the rank of values: out[b, k] = values[b, idx[b, k]].
    return torch.gather(values, 1, idx)


def reorder_steps(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    # gather along axis 1 reads out[b, t, d] = x[b, index[b, t, d], d], so the index
    # needs an entry for every feature d: (B, T) -> (B, T, 1) -> (B, T, D), with the
    # same step number repeated along the feature axis.
    index = perm[:, :, None].expand(-1, -1, x.shape[2])
    return torch.gather(x, 1, index)
