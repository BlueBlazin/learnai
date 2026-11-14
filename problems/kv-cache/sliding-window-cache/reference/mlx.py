import mlx.core as mx

Cache = tuple[mx.array, mx.array, int]


def rotating_write(
    cache: Cache, k_new: mx.array, v_new: mx.array, max_size: int, keep: int
) -> tuple[Cache, mx.array, mx.array]:
    k, v, offset = cache
    window = max_size - keep
    end = offset + k_new.shape[2]
    # Token numbers the cache holds once this write is done, oldest first: the sinks
    # 0 .. keep - 1, then the `window` most recent tokens (none while end <= keep).
    sinks = mx.arange(min(keep, end))
    recent = mx.arange(max(keep, end - window), max(keep, end))
    live = mx.concatenate((sinks, recent))
    # Where each of them lives: a sink in its own slot, the others on a ring of `window`
    # slots behind the sinks. A token's slot never changes.
    slots = mx.where(live < keep, live, keep + (live - keep) % window)
    # The new tokens are the tail of `live`. New tokens that are already out of the
    # window (a chunk longer than the window) are not in it, and are never written.
    fresh = int((live >= offset).sum().item())
    k[:, :, slots[-fresh:]] = k_new[:, :, live[-fresh:] - offset]
    v[:, :, slots[-fresh:]] = v_new[:, :, live[-fresh:] - offset]
    # Reading the slots in order of token number gives the entries in temporal order.
    return (k, v, end), k[:, :, slots], v[:, :, slots]
