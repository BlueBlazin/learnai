# BUG: the dot products are summed over the groups first and scaled once per row with the row's mean scale, as if all groups of a row shared one scale.
import torch


def qmatmul(
    x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, group_size: int
) -> torch.Tensor:
    out_dim, in_dim = codes.shape
    n_groups = in_dim // group_size
    # Cut the inputs and the codes into the same consecutive groups along the input axis.
    xg = x.reshape(*x.shape[:-1], n_groups, group_size)  # (..., G, g)
    cg = codes.reshape(out_dim, n_groups, group_size).to(x.dtype)  # (O, G, g)
    # One short dot product of inputs with raw codes per (output, group).
    dots = torch.einsum("...gk,ogk->...og", xg, cg)  # (..., O, G)
    # The bias of a group multiplies the sum of that group's inputs, whatever the codes are.
    sums = xg.sum(dim=-1)  # (..., G)
    return dots.sum(dim=-1) * scale.mean(dim=-1) + sums @ bias.T
