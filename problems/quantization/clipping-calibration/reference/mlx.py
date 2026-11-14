import mlx.core as mx


def best_clip(samples: mx.array, bits: int, candidates: mx.array) -> mx.array:
    qmax = 2 ** (bits - 1) - 1  # symmetric grid: codes -qmax ... qmax
    x = samples.reshape(-1)  # the calibration set is pooled: one range for all of it
    limit = candidates * mx.abs(x).max()  # (K,) clipping thresholds
    scale = (limit / qmax)[:, None]  # (K, 1): one grid per candidate
    # Quantize all samples on every candidate grid at once: (K, N).
    q = mx.clip(mx.round(x / scale), -qmax, qmax)
    # The error counts both effects: rounding inside the range and clipping outside it.
    mse = ((q * scale - x) ** 2).mean(axis=-1)  # (K,)
    # Of the candidates that reach the minimum, return the largest range.
    return mx.where(mse == mse.min(), candidates, mx.zeros_like(candidates)).max()
