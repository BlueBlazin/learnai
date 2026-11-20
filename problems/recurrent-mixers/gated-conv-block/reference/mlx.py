import mlx.core as mx


def conv_block(
    x: mx.array,
    w_in: mx.array,
    w_conv: mx.array,
    w_out: mx.array,
    state: mx.array | None,
) -> tuple[mx.array, mx.array]:
    steps, kernel = x.shape[0], w_conv.shape[1]
    # Three equal parts, in this order: input gate, output gate, signal.
    B, C, h = mx.split(x @ w_in.T, 3, axis=-1)
    u = B * h  # what the convolution reads, and what the state remembers
    if state is None:
        state = mx.zeros((kernel - 1, u.shape[1]), dtype=u.dtype)
    padded = mx.concatenate((state, u), axis=0)  # (L - 1 + T, D)
    # Output t reads rows t .. t + L - 1 of padded: tap j meets row t + j, so the
    # last tap multiplies the current token. No flip: this is a cross-correlation.
    conv = sum(padded[j : j + steps] * w_conv[:, j] for j in range(kernel))
    y = (C * conv) @ w_out.T
    return y, padded[steps:]  # the last L - 1 rows of u, also when T < L - 1
