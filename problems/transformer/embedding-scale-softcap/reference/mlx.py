import mlx.core as mx


def embed(tokens: mx.array, table: mx.array, scale: float) -> mx.array:
    # Scale the rows that were looked up, not the table: a tied head reads the same
    # matrix and must see it unscaled.
    return table[tokens] * scale


def finalize_logits(
    h: mx.array,
    w_out: mx.array,
    logit_scale: float,
    softcap: float | None = None,
) -> mx.array:
    # w_out is (V, d), so the projection is h @ W^T: (..., d) -> (..., V).
    logits = h @ w_out.T / logit_scale  # logit_scale is a divisor
    if softcap is not None:
        # Smoothly squash into (-softcap, softcap); near zero this is the identity.
        logits = softcap * mx.tanh(logits / softcap)
    return logits
