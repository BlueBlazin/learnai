import mlx.core as mx

from whetstone.kit import ops


def attend_step(
    q_new: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    cache: tuple[mx.array, mx.array] | None,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    if cache is None:
        k, v = k_new, v_new
    else:
        # Append along the sequence axis (2), old entries first. `concatenate` builds
        # new arrays, so the cache that was passed in is left as it was.
        k = mx.concatenate((cache[0], k_new), axis=2)
        v = mx.concatenate((cache[1], v_new), axis=2)
    new, total = q_new.shape[2], k.shape[2]
    past = total - new
    # New token i sits at absolute position past + i and may see keys 0 .. past + i:
    # every cached key, and the new keys up to and including its own.
    visible = mx.arange(total)[None, :] <= past + mx.arange(new)[:, None]  # (T, S + T)
    return ops.attention(q_new, k, v, visible), (k, v)
