import mlx.core as mx


def ssd_step(
    h: mx.array,
    x: mx.array,
    dt_raw: mx.array,
    B: mx.array,
    C: mx.array,
    A_log: mx.array,
    D: mx.array,
    dt_bias: mx.array,
    dt_limit: tuple[float, float],
) -> tuple[mx.array, mx.array]:
    """One token of a Mamba-2 state-space layer, for H heads in G groups.

    h: float32, shape (H, P, N). The state before the token.
    x: float32, shape (H, P). The token's input, split into heads.
    dt_raw: float32, shape (H,). B, C: float32, shape (G, N); group j serves the
        H / G consecutive heads j * H / G, ..., (j + 1) * H / G - 1.
    A_log, D, dt_bias: float32, shape (H,). Parameters.
    dt_limit: (low, high), Python floats; high may be float("inf").
    Returns (y, h): y float32 of shape (H, P), read from the updated state, and the
    updated state, float32 of shape (H, P, N).
    """
    raise NotImplementedError


def ssd_matrix(dt: mx.array, B: mx.array, C: mx.array, A_log: mx.array) -> mx.array:
    """The matrix that maps the inputs of T tokens to the outputs of the same layer.

    dt: float32, shape (T, H). Step sizes, already positive and clamped.
    B, C: float32, shape (T, G, N). A_log: float32, shape (H,).
    Returns float32 of shape (H, T, T), lower triangular in its last two axes:
    M[h, t, s] = (C_t . B_s) * exp(-exp(A_log_h) * (dt_{s+1} + ... + dt_t)) * dt_s.
    """
    raise NotImplementedError
