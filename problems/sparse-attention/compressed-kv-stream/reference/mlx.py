import mlx.core as mx

State = tuple[mx.array, mx.array, int]


def compress(
    kv: mx.array,
    gate: mx.array,
    position_bias: mx.array,
    gain: mx.array,
    state: State,
    rate: int,
) -> tuple[mx.array, State]:
    buf_kv, buf_gate, count = state
    # The rows that wait from earlier calls come first: they are older.
    kv, gate = mx.concatenate((buf_kv, kv)), mx.concatenate((buf_gate, gate))
    windows = kv.shape[0] // rate  # complete windows now available; may be 0
    usable, dim = windows * rate, kv.shape[1]

    # (windows, rate, d): row j of a window is the stream row at offset j in it, so the
    # position bias lines up with the window however the stream was cut.
    win_kv = kv[:usable].reshape(windows, rate, dim)
    win_gate = gate[:usable].reshape(windows, rate, dim) + position_bias
    # One softmax per window and channel, over the rate rows of the window.
    pooled = (mx.softmax(win_gate, axis=1) * win_kv).sum(axis=1)  # (windows, d)
    entries = gain * pooled * mx.rsqrt((pooled**2).mean(axis=-1, keepdims=True) + 1e-6)

    # The unfinished window waits, as received (no bias added yet).
    return entries, (kv[usable:], gate[usable:], count + windows)


def visible_entries(t: int, rate: int) -> int:
    # Entry w ends at position (w + 1) * rate - 1; it is visible from that position on.
    return (t + 1) // rate
