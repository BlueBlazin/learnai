import mlx.core as mx


def relative_positions(T: int) -> mx.array:
    pos = mx.arange(T)
    # A column of i against a row of j: (T, 1) - (1, T) broadcasts to (T, T).
    return pos[:, None] - pos[None, :]


def scale_rows(x: mx.array, s: mx.array) -> mx.array:
    # (B,) lines up with the last axis of x. (B, 1, 1) lines up with (B, T, D).
    return x * s[:, None, None]


def pairwise_sq_dist(a: mx.array, b: mx.array) -> mx.array:
    # (N, 1, D) - (1, M, D) broadcasts to (N, M, D): every point of a against every
    # point of b. The feature axis is then summed away.
    diff = a[:, None, :] - b[None, :, :]
    return mx.sum(diff * diff, axis=-1)
