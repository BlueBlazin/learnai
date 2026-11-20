import math

import mlx.core as mx


def yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    original_max_positions: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    truncate: bool = True,
) -> mx.array:
    """The dim / 2 rotary inverse frequencies, rescaled by YaRN.

    dim: even int, the number of rotated channels. base: float > 1. factor: float >= 1.
    original_max_positions: int, the context length the model was trained with.
    beta_fast, beta_slow: full turns over that length above which a pair keeps its
        frequency and below which it is divided by factor.
    truncate: floor the lower end of the ramp and ceil the upper end.
    Returns float32, shape (dim / 2,).
    """
    raise NotImplementedError


def yarn_tables(
    positions: mx.array, inv_freq: mx.array, factor: float
) -> tuple[mx.array, mx.array]:
    """cos and sin of positions[t] * inv_freq[i], both times 0.1 * ln(factor) + 1.

    positions: integer tensor, shape (T,). inv_freq: float32, shape (P,).
    Returns (cos, sin), each float32 of shape (T, P).
    """
    raise NotImplementedError
