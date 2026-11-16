import mlx.core as mx

from learnai.kit import ops


def rope_positions(cache_lengths: mx.array, n_new: int) -> mx.array:
    # Row b already holds cache_lengths[b] tokens, at positions 0 .. cache_lengths[b] - 1,
    # so its new tokens continue at cache_lengths[b], cache_lengths[b] + 1, ...
    return cache_lengths[:, None] + mx.arange(n_new)  # (B, 1) + (n_new,) -> (B, n_new)


def rotate_new(
    q: mx.array, k: mx.array, positions: mx.array, theta: float
) -> tuple[mx.array, mx.array]:
    # Only the new tokens are rotated, each for its own absolute position. The cached
    # keys were rotated when they were new and are never touched again.
    return ops.rope(q, positions, theta), ops.rope(k, positions, theta)
