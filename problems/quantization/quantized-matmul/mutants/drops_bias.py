# BUG: the bias term is dropped, so the result is x @ (codes * scale).T.
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
    return (dots * scale).sum(dim=-1)
