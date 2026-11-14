import mlx.core as mx

from whetstone.kit import ops

Cache = tuple[mx.array, mx.array, int]


def cache_init(capacity: int, batch: int, heads: int, dim: int) -> Cache:
    shape = (batch, heads, capacity, dim)
    return mx.zeros(shape), mx.zeros(shape), 0


def cache_write(cache: Cache, k_new: mx.array, v_new: mx.array) -> Cache:
    k, v, length = cache
    end = length + k_new.shape[2]
    if end > k.shape[2]:
        raise ValueError(f"writing {k_new.shape[2]} tokens at length {length} exceeds {k.shape[2]}")
    # Fill slots length .. end - 1 of the existing buffers; the shape stays the same and
    # no other slot is touched. The caller continues with the returned cache.
    k[:, :, length:end] = k_new
    v[:, :, length:end] = v_new
    return k, v, end


def cache_attend(q: mx.array, cache: Cache) -> mx.array:
    k, v, length = cache
    new = q.shape[2]
    # Only the first `length` slots hold tokens. Slicing them off keeps whatever the
    # tail contains out of the arithmetic altogether.
    k, v = k[:, :, :length], v[:, :, :length]
    # The queries belong to the last `new` tokens: query i is at position
    # length - new + i and sees the slots up to and including its own.
    visible = mx.arange(length)[None, :] <= (length - new) + mx.arange(new)[:, None]
    return ops.attention(q, k, v, visible)
