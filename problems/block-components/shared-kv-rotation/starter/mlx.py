import math

import mlx.core as mx


def shared_kv_attention(
    q: mx.array,
    kv: mx.array,
    pos_q: mx.array,
    pos_k: mx.array,
    mask: mx.array,
    rope_dim: int,
    base: float,
) -> mx.array:
    """Attention in which one tensor is both key and value, with a rotary slice.

    q: float32, shape (H, T, d). One query per head and token.
    kv: float32, shape (S, d). Row s is the key and the value of token s, for every head.
    pos_q: integer tensor, shape (T,). pos_k: integer tensor, shape (S,).
    mask: bool, shape (T, S). True where query t may attend to row s.
    rope_dim: even int, 2 <= rope_dim <= d. The trailing rope_dim channels rotate.
    base: float > 1, the RoPE base of the rotary slice.
    Returns float32, shape (H, T, d).
    """
    raise NotImplementedError
