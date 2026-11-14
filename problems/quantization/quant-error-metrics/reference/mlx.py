import mlx.core as mx


def mse(a: mx.array, b: mx.array) -> mx.array:
    # One number for the whole tensor: the mean runs over every element.
    return mx.mean((a - b) ** 2)


def sqnr_db(ref: mx.array, approx: mx.array) -> mx.array:
    signal = mx.sum(ref**2)
    noise = mx.sum((ref - approx) ** 2)
    # A ratio of powers (squared quantities) takes 10 log10. Zero noise gives +inf.
    return 10.0 * mx.log10(signal / noise)


def max_abs_error(a: mx.array, b: mx.array) -> mx.array:
    return mx.max(mx.abs(a - b))


def output_error(w: mx.array, w_hat: mx.array, x: mx.array) -> mx.array:
    # A linear layer computes x w^T: one row of outputs per row of x.
    y = x @ w.T
    y_hat = x @ w_hat.T
    # Frobenius norms over the whole output, normalized by the exact output.
    return mx.sqrt(mx.sum((y - y_hat) ** 2)) / mx.sqrt(mx.sum(y**2))
