# BUG: each row is RMS-normalized before the weighted sum instead of the sum being normalized, so the entry no longer has unit root mean square before the gain.
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
    normed = win_kv * torch.rsqrt(win_kv.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    entries = gain * (torch.softmax(win_gate, dim=1) * normed).sum(dim=1)
    return entries, (kv[usable:], gate[usable:], count + windows)


def visible_entries(t: int, rate: int) -> int:
    return (t + 1) // rate
