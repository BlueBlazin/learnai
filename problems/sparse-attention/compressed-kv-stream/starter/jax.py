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
    """Add n rows to a stream and emit one entry per window of `rate` rows completed.

    kv, gate: float32, shape (n, d), n >= 1. The next n rows of the stream.
    position_bias: float32, shape (rate, d). Row j is added to the gate of the row at
        offset j of its window.
    gain: float32, shape (d,). The weight of the RMS norm.
    state: (buf_kv, buf_gate, count). The rows of the unfinished window, float32 of
        shape (r, d) each with 0 <= r < rate, as they were received, and the number of
        entries emitted so far.
    rate: int >= 1. The window length.
    Returns (entries, state): entries is float32 of shape (m, d), one row per window
    completed by this call, oldest first (m may be 0); the new state holds the rows
    that are left and count + m.
    """
    raise NotImplementedError


def visible_entries(t: int, rate: int) -> int:
    """How many entries the query at position t >= 0 may attend to."""
    raise NotImplementedError
