import mlx.core as mx

Cache = tuple[mx.array, mx.array, int]  # (k, v, offset)


def rotating_write(
    cache: Cache, k_new: mx.array, v_new: mx.array, max_size: int, keep: int
) -> tuple[Cache, mx.array, mx.array]:
    """Write T new tokens into a rotating cache and return its contents in temporal order.

    cache: (k, v, offset). k, v: float32 buffers of shape (B, H, max_size, D). offset: the
        number of tokens written so far (a Python int; it may exceed max_size).
    k_new, v_new: float32, shape (B, H, T, D), T >= 1.
    max_size: slots in the buffer. keep: 0 <= keep < max_size; the first `keep` tokens
        are never evicted. Token n lives in slot n if n < keep, and otherwise in slot
        keep + (n - keep) % (max_size - keep).
    Returns ((k, v, offset + T), keys, values): the buffers keep their shape; keys and
    values are float32 of shape (B, H, min(offset + T, max_size), D), the cached tokens
    from oldest to newest.
    """
    raise NotImplementedError
