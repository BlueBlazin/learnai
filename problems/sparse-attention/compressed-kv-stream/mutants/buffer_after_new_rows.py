# BUG: the waiting rows are concatenated after the new rows instead of before them, so a window that spans two calls has its rows in the wrong order.
import torch

State = tuple[torch.Tensor, torch.Tensor, int]


def compress(
    kv: torch.Tensor,
    gate: torch.Tensor,
    position_bias: torch.Tensor,
    gain: torch.Tensor,
    state: State,
    rate: int,
) -> tuple[torch.Tensor, State]:
    buf_kv, buf_gate, count = state
    kv, gate = torch.cat((kv, buf_kv)), torch.cat((gate, buf_gate))
    windows = kv.shape[0] // rate
    usable, dim = windows * rate, kv.shape[1]
    win_kv = kv[:usable].reshape(windows, rate, dim)
    win_gate = gate[:usable].reshape(windows, rate, dim) + position_bias
    pooled = (torch.softmax(win_gate, dim=1) * win_kv).sum(dim=1)
    entries = gain * pooled * torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return entries, (kv[usable:], gate[usable:], count + windows)


def visible_entries(t: int, rate: int) -> int:
    return (t + 1) // rate
