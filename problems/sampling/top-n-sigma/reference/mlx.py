import mlx.core as mx


def top_n_sigma_filter(logits: mx.array, n: float) -> mx.array:
    if n <= 0:
        return logits
    # The statistics are over the finite logits only: a masked entry is not a small
    # number, it is absent. Zero it out of the sums and divide by the finite count.
    finite = mx.isfinite(logits)
    count = mx.sum(finite, axis=-1, keepdims=True)
    mean = mx.sum(mx.where(finite, logits, 0.0), axis=-1, keepdims=True) / count
    squares = mx.where(finite, (logits - mean) ** 2, 0.0)
    std = mx.sqrt(mx.sum(squares, axis=-1, keepdims=True) / count)  # population: divide by |F|
    # A band of width n * std below the row maximum. `>=` keeps the maximum itself when
    # std is 0, and -inf entries fail the comparison and stay -inf.
    cutoff = mx.max(logits, axis=-1, keepdims=True) - n * std
    return mx.where(logits >= cutoff, logits, -mx.inf)
