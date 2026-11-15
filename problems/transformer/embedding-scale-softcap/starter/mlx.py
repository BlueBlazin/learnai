import mlx.core as mx


def embed(tokens: mx.array, table: mx.array, scale: float) -> mx.array:
    """Look up token embeddings and multiply them by the model's embedding multiplier.

    tokens: integer tensor of any shape (...), ids in [0, V).
    table: float32, shape (V, d). The embedding matrix; it must not be modified.
    scale: float. The embedding multiplier (Granite 4.0 350M: 12.0; Gemma: sqrt(d)).
    Returns float32, shape (..., d).
    """
    raise NotImplementedError


def finalize_logits(
    h: mx.array,
    w_out: mx.array,
    logit_scale: float,
    softcap: float | None = None,
) -> mx.array:
    """Project final hidden states to logits, divide, and optionally soft-cap.

    h: float32, shape (..., d). Hidden states after the final norm.
    w_out: float32, shape (V, d). The output head in (out, in) layout.
    logit_scale: float > 0. The logits are divided by it (Granite 4.0 350M: 4.0).
    softcap: None, or a float > 0: logits become softcap * tanh(logits / softcap),
        applied after the division (Gemma 2: 30.0).
    Returns float32, shape (..., V).
    """
    raise NotImplementedError
