# BUG: the RMS norm divides by the root mean square without the 1e-6 under the root: small entries come out too large, and a window of zeros gives NaN.
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
    kv, gate = torch.cat((buf_kv, kv)), torch.cat((buf_gate, gate))
    windows = kv.shape[0] // rate
    usable, dim = windows * rate, kv.shape[1]
    win_kv = kv[:usable].reshape(windows, rate, dim)
    win_gate = gate[:usable].reshape(windows, rate, dim) + position_bias
    pooled = (torch.softmax(win_gate, dim=1) * win_kv).sum(dim=1)
    entries = gain * pooled * torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True))
    return entries, (kv[usable:], gate[usable:], count + windows)


def visible_entries(t: int, rate: int) -> int:
    return (t + 1) // rate
