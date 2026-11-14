# BUG: each bias is multiplied by group_size instead of by the sum of its group's inputs, which is only right when every input is 1.
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
    # Every group contributes its bias once per element.
    return (dots * scale).sum(dim=-1) + group_size * bias.sum(dim=-1)
