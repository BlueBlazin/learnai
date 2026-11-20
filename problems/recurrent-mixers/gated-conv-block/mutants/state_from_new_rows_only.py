# BUG: the new state is cut from the rows of this call alone, padded with zeros, so a call with fewer than L - 1 tokens forgets the rows saved before it.
import torch


def conv_block(
    x: torch.Tensor,
    w_in: torch.Tensor,
    w_conv: torch.Tensor,
    w_out: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps, kernel = x.shape[0], w_conv.shape[1]
    # Three equal parts, in this order: input gate, output gate, signal.
    B, C, h = torch.chunk(x @ w_in.T, 3, dim=-1)
    u = B * h  # what the convolution reads, and what the state remembers
    if state is None:
        state = torch.zeros(kernel - 1, u.shape[1])
    padded = torch.cat((state, u), dim=0)  # (L - 1 + T, D)
    # Output t reads rows t .. t + L - 1 of padded: tap j meets row t + j, so the
    # last tap multiplies the current token. No flip: this is a cross-correlation.
    conv = sum(padded[j : j + steps] * w_conv[:, j] for j in range(kernel))
    y = (C * conv) @ w_out.T
    fresh = torch.cat((torch.zeros_like(state), u), dim=0)
    return y, fresh[steps:]  # only rows of this call
