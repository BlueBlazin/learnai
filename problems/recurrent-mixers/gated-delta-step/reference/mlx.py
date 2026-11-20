import mlx.core as mx


def gdn_gates(
    a: mx.array, b: mx.array, A_log: mx.array, dt_bias: mx.array
) -> tuple[mx.array, mx.array]:
    # softplus is positive and exp(A_log) is positive, so the exponent is negative
    # and g lies in (0, 1): the memory can fade, never grow.
    # logaddexp(x, 0) is softplus without the rounding of log(1 + exp(x)).
    g = mx.exp(-mx.exp(A_log) * mx.logaddexp(a + dt_bias, 0.0))
    beta = mx.sigmoid(b)
    return g, beta


def gdn_step(
    S: mx.array, q: mx.array, k: mx.array, v: mx.array, g: mx.array, beta: mx.array
) -> tuple[mx.array, mx.array]:
    # 1. Decay: every stored association fades by g, one factor per head.
    S = S * g[:, None, None]
    # 2. Write. S k is what the decayed memory returns for this key; only the
    #    difference to v is stored, along the direction k.
    predicted = mx.einsum("hvk,hk->hv", S, k)
    delta = beta[:, None] * (v - predicted)
    S = S + delta[:, :, None] * k[:, None, :]  # outer product, (H, Dv, Dk)
    # 3. Read from the updated memory: the token sees its own write.
    y = mx.einsum("hvk,hk->hv", S, q)
    return y, S
