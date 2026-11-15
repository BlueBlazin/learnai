# BUG: the votes are pooled over all L columns and cut to the prefix afterwards, so the attention the window pays to itself leaks into the last prefix positions.
import torch
import torch.nn.functional as F


def snapkv_select(attn: torch.Tensor, window: int, budget: int, kernel: int) -> torch.Tensor:
    b, h, length, _ = attn.shape
    if length <= budget:
        return torch.arange(length).expand(b, h, length)
    prefix = length - window
    # Vote: what each prefix position receives from the queries of the observation
    # window, per head. Rows before the window and the window's own columns do not count.
    vote = attn[:, :, prefix:, :].sum(dim=2)  # (B, H, L)
    # Max-pool along the sequence, stride 1: a position gets the best vote within
    # kernel // 2 of it. Padding with -inf keeps the edges inside the prefix.
    half = kernel // 2
    padded = F.pad(vote, (half, half), value=float("-inf"))
    pooled = torch.stack([padded[..., o : o + length] for o in range(kernel)]).amax(dim=0)
    pooled = pooled[..., :prefix]
    # A stable descending sort ranks equal pooled votes by position, lowest first.
    ranked = torch.argsort(pooled, dim=-1, descending=True, stable=True)
    chosen = torch.sort(ranked[..., : budget - window], dim=-1).values
    tail = torch.arange(prefix, length).expand(b, h, window)
    return torch.cat((chosen, tail), dim=-1)
