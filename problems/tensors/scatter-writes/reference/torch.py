import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    # scatter along axis 1 writes out[b, index[b, j]] = 1. The index needs as many
    # axes as the tensor written into: (B,) -> (B, 1).
    out = torch.zeros(ids.shape[0], V, dtype=torch.float32)
    return out.scatter(1, ids[:, None], 1.0)


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    # Send a 1 to column ids[b, t] for every t. scatter_add accumulates the values
    # that land on one entry; scatter would keep a single 1 for a repeated token.
    counts = torch.zeros(ids.shape[0], V, dtype=torch.int64)
    return counts.scatter_add(1, ids, torch.ones_like(ids))


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    # scatter (no trailing underscore) returns a new tensor: out[b, idx[b, k]] =
    # values[b, k], and every other entry is copied from x.
    return x.scatter(1, idx, values)
