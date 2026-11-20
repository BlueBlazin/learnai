# BUG: the state tensor the caller passed in is shifted and overwritten in place, row by row, instead of a new state being returned.
import torch


def conv_block(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_conv: torch.Tensor,
    w_out: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = w_conv.shape[1]
    B, C, h = torch.chunk(x @ w_in.T, 3, dim=-1)
    u = B * h
    if state is None:
        state = torch.zeros(kernel - 1, u.shape[1])
    rows = []
    for t in range(x.shape[0]):
        window = torch.cat((state, u[t : t + 1]), dim=0)  # (L, D)
        rows.append((window * w_conv.T).sum(dim=0))
        state.copy_(window[1:])  # in place: the caller's tensor changes too
    y = (C * torch.stack(rows)) @ w_out.T
    return y, state
