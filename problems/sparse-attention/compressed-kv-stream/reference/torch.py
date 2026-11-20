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
    # The rows that wait from earlier calls come first: they are older.
    kv, gate = torch.cat((buf_kv, kv)), torch.cat((buf_gate, gate))
    windows = kv.shape[0] // rate  # complete windows now available; may be 0
    usable, dim = windows * rate, kv.shape[1]

    # (windows, rate, d): row j of a window is the stream row at offset j in it, so the
    # position bias lines up with the window however the stream was cut.
    win_kv = kv[:usable].reshape(windows, rate, dim)
    win_gate = gate[:usable].reshape(windows, rate, dim) + position_bias
    # One softmax per window and channel, over the rate rows of the window.
    pooled = (torch.softmax(win_gate, dim=1) * win_kv).sum(dim=1)  # (windows, d)
    entries = gain * pooled * torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    # The unfinished window waits, as received (no bias added yet).
    return entries, (kv[usable:], gate[usable:], count + windows)


def visible_entries(t: int, rate: int) -> int:
    # Entry w ends at position (w + 1) * rate - 1; it is visible from that position on.
    return (t + 1) // rate
