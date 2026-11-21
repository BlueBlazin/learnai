import mlx.core as mx

NORM_EPS = 1e-6  # rms_norm_eps of DeepSeek-V4; not the `eps` argument


def hc_weights(
    streams: mx.array,
    fn: mx.array,
    base: mx.array,
    scale: mx.array,
    iters: int,
    eps: float,
) -> tuple[mx.array, mx.array, mx.array]:
    steps, n, dim = streams.shape
    # One vector per token: all streams end to end, normalized as a whole, no gain.
    flat = streams.reshape(steps, n * dim)
    flat = flat * mx.rsqrt(mx.mean(flat * flat, axis=-1, keepdims=True) + NORM_EPS)
    # One linear map gives the raw values of all three weights: N, N, and N * N of them.
    raw = flat @ fn.T
    pre = mx.sigmoid(raw[:, :n] * scale[0] + base[:n]) + eps
    post = 2.0 * mx.sigmoid(raw[:, n : 2 * n] * scale[1] + base[n : 2 * n])
    logits = (raw[:, 2 * n :] * scale[2] + base[2 * n :]).reshape(steps, n, n)
    comb = mx.softmax(logits, axis=-1) + eps

    # Sinkhorn-Knopp: columns once, then (iters - 1) times rows and columns again.
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)

    collapsed = (pre[:, :, None] * streams).sum(axis=1)
    return collapsed, post, comb


def hc_merge(streams: mx.array, y: mx.array, post: mx.array, comb: mx.array) -> mx.array:
    # Output stream n = post[n] * y + sum_j comb[j, n] * streams[j]: comb is used transposed.
    return post[:, :, None] * y[:, None, :] + mx.swapaxes(comb, -1, -2) @ streams
