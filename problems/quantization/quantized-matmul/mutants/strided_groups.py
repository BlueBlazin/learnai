# BUG: x is cut into groups with reshape(..., group_size, n_groups) and a transpose, so group g holds every n_groups-th input instead of group_size consecutive ones, misaligned with the codes.
import torch


def qmatmul(
    x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, group_size: int
) -> torch.Tensor:
    out_dim, in_dim = codes.shape
    n_groups = in_dim // group_size
    # Cut the inputs and the codes into the same consecutive groups along the input axis.
    xg = x.reshape(*x.shape[:-1], group_size, n_groups).transpose(-1, -2)  # (..., G, g)
    cg = codes.reshape(out_dim, n_groups, group_size).to(x.dtype)  # (O, G, g)
    # One short dot product of inputs with raw codes per (output, group).
    dots = torch.einsum("...gk,ogk->...og", xg, cg)  # (..., O, G)
    # The bias of a group multiplies the sum of that group's inputs, whatever the codes are.
    sums = xg.sum(dim=-1)  # (..., G)
    return (dots * scale).sum(dim=-1) + sums @ bias.T
