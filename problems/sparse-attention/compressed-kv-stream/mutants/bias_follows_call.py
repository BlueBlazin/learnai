# BUG: the position bias is added to the new rows as they arrive, indexed from the first row of the call instead of by their offset in the window, so a cut inside a window shifts the bias (and the waiting gates are stored with the bias already added).
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
    offsets = torch.arange(gate.shape[0]) % rate
    kv, gate = torch.cat((buf_kv, kv)), torch.cat((buf_gate, gate + position_bias[offsets]))
    windows = kv.shape[0] // rate
    usable, dim = windows * rate, kv.shape[1]
    win_kv = kv[:usable].reshape(windows, rate, dim)
    win_gate = gate[:usable].reshape(windows, rate, dim)
    pooled = (torch.softmax(win_gate, dim=1) * win_kv).sum(dim=1)
    entries = gain * pooled * torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return entries, (kv[usable:], gate[usable:], count + windows)


def visible_entries(t: int, rate: int) -> int:
    return (t + 1) // rate
