import mlx.core as mx


def conv_block(
    x: mx.array,
    w_in: mx.array,
    w_conv: mx.array,
    w_out: mx.array,
    state: mx.array | None,
) -> tuple[mx.array, mx.array]:
    """A gated causal convolution over T new tokens, continuing from `state`.

    x: float32, shape (T, D), T >= 1.
    w_in: float32, shape (3 * D, D). w_out: float32, shape (D, D). Applied as x @ w.T.
    w_conv: float32, shape (D, L). One kernel of L taps per channel; the last tap
        multiplies the current token.
    state: None for a new sequence, or float32 of shape (L - 1, D): the last L - 1
        rows of u = B * h from the tokens before x.
    Returns (y, state): y float32 of shape (T, D) and the new state, float32 of
    shape (L - 1, D).
    """
    raise NotImplementedError
