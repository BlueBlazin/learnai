import jax
import jax.numpy as jnp

State = tuple[jax.Array, jax.Array, int]


def compress(
    kv: jax.Array,
    gate: jax.Array,
    position_bias: jax.Array,
    gain: jax.Array,
    state: State,
    rate: int,
) -> tuple[jax.Array, State]:
    buf_kv, buf_gate, count = state
    # The rows that wait from earlier calls come first: they are older.
    kv, gate = jnp.concatenate((buf_kv, kv)), jnp.concatenate((buf_gate, gate))
    windows = kv.shape[0] // rate  # complete windows now available; may be 0
    usable, dim = windows * rate, kv.shape[1]

    # (windows, rate, d): row j of a window is the stream row at offset j in it, so the
    # position bias lines up with the window however the stream was cut.
    win_kv = kv[:usable].reshape(windows, rate, dim)
    win_gate = gate[:usable].reshape(windows, rate, dim) + position_bias
    # One softmax per window and channel, over the rate rows of the window.
    pooled = (jax.nn.softmax(win_gate, axis=1) * win_kv).sum(axis=1)  # (windows, d)
    entries = gain * pooled * jax.lax.rsqrt((pooled**2).mean(axis=-1, keepdims=True) + 1e-6)

    # The unfinished window waits, as received (no bias added yet).
    return entries, (kv[usable:], gate[usable:], count + windows)


def visible_entries(t: int, rate: int) -> int:
    # Entry w ends at position (w + 1) * rate - 1; it is visible from that position on.
    return (t + 1) // rate
