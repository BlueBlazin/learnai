# BUG: the fused weight is split along the input axis (columns) instead of the output axis (rows).
import torch
import torch.nn.functional as F


def split_fused(w: torch.Tensor, parts: list[int]) -> list[torch.Tensor]:
    # Weights are (out, in): the projections are stacked along the output axis, so
    # each one is a block of consecutive rows. `parts` holds sizes, not offsets.
    return list(torch.split(w, parts, dim=1))


def fuse(ws: list[torch.Tensor]) -> torch.Tensor:
    # Stacking rows concatenates the outputs: x @ fuse(ws)^T = cat(x @ w^T for w in ws).
    return torch.cat(ws, dim=0)


def swiglu_fused(x: torch.Tensor, w_in: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    # One matrix product computes both branches; the first half of the features is
    # the gate (it gets the SiLU), the second half is the up branch.
    gate, up = (x @ w_in.T).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w_down.T
