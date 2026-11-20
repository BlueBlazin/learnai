# BUG: the rows left over at the end of a call are compressed into an entry at once, with the bias of their offsets, instead of waiting for the rest of their window.
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
    entries = []
    for start in range(0, kv.shape[0], rate):
        win_kv, win_gate = kv[start : start + rate], gate[start : start + rate]
        weights = torch.softmax(win_gate + position_bias[: win_kv.shape[0]], dim=0)
        pooled = (weights * win_kv).sum(dim=0)
        entries.append(gain * pooled * torch.rsqrt(pooled.pow(2).mean() + 1e-6))
    return torch.stack(entries), (kv[:0], gate[:0], count + len(entries))


def visible_entries(t: int, rate: int) -> int:
    return (t + 1) // rate
