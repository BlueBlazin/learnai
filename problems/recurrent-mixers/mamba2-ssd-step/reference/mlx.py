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
    # The step size of each head: positive, then clamped to the configured range.
    # logaddexp(x, 0) is softplus without the rounding of log(1 + exp(x)).
    dt = mx.clip(mx.logaddexp(dt_raw + dt_bias, 0.0), dt_limit[0], dt_limit[1])
    # Group j serves the heads j * r .. j * r + r - 1: repeat, do not tile.
    reps = h.shape[0] // B.shape[0]
    B = mx.repeat(B, reps, axis=0)  # (H, N)
    C = mx.repeat(C, reps, axis=0)
    # h <- h * exp(dt * A) + dt * x B^T, with A = -exp(A_log) < 0.
    decay = mx.exp(-dt * mx.exp(A_log))
    h = h * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[:, None, :]
    # Read the updated state, plus the skip connection.
    y = mx.einsum("hpn,hn->hp", h, C) + D[:, None] * x
    return y, h


def ssd_matrix(dt: mx.array, B: mx.array, C: mx.array, A_log: mx.array) -> mx.array:
    steps, heads = dt.shape
    reps = heads // B.shape[1]
    B = mx.repeat(B, reps, axis=1)  # (T, H, N)
    C = mx.repeat(C, reps, axis=1)
    scores = mx.einsum("thn,shn->hts", C, B)  # C_t . B_s
    # exp(sum of dt_r * A for s < r <= t) = exp(c_t - c_s), c the running sum of dt * A.
    c = mx.cumsum(-dt * mx.exp(A_log), axis=0).T  # (H, T)
    lower = mx.tril(mx.ones((steps, steps))) > 0
    decay = mx.exp(mx.where(lower, c[:, :, None] - c[:, None, :], -mx.inf))
    return scores * decay * dt.T[:, None, :]  # the input at s enters scaled by dt_s
