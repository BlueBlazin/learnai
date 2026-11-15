import mlx.core as mx

EPS = 1e-12


def _normalize(x: mx.array) -> mx.array:
    # Divide by the L2 norm, but never by less than EPS: a zero vector stays zero
    # instead of becoming 0 / 0.
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), EPS)


def pool(states: mx.array, mask: mx.array, mode: str) -> mx.array:
    batch, length, _ = states.shape
    positions = mx.arange(length)
    if mode == "mean":
        # Zero the padded states, then divide by the number of real tokens, not by T.
        weights = mask.astype(states.dtype)[..., None]  # (B, T, 1)
        pooled = (states * weights).sum(axis=1) / weights.sum(axis=1)
    else:
        if mode == "cls":
            # The first real token: the smallest position whose mask is set.
            index = mx.where(mask != 0, positions, length).min(axis=1)
        else:
            # The last real token: the largest position whose mask is set. With right
            # padding that is lengths - 1; with left padding it is T - 1.
            index = (positions * (mask != 0)).max(axis=1)
        pooled = states[mx.arange(batch), index]  # (B, D)
    return _normalize(pooled)


def truncate_matryoshka(e: mx.array, dim: int) -> mx.array:
    # A prefix of a unit vector is shorter than 1: slice, then normalize again.
    return _normalize(e[..., :dim])
