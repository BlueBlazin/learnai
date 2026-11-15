import mlx.core as mx


def multi_draft_step(
    p: mx.array,
    q: mx.array,
    candidates: mx.array,
    u_accept: mx.array,
    u_resample: float,
) -> tuple[int, int]:
    """One position of multi-draft speculative sampling (recursive rejection sampling).

    p: float32 tensor of shape (V,), the target model's distribution.
    q: float32 tensor of shape (V,), the draft model's distribution.
    candidates: integer tensor of shape (n,), tokens drawn independently from q.
    u_accept: float32 tensor of shape (n,), uniforms in [0, 1); u_accept[i] decides
        candidate i, tested against the residual left by the rejections before it.
    u_resample: uniform float in [0, 1) for the inverse-CDF sample from the final
        residual; used only when every candidate is rejected.
    Returns (token, n_rejected) as Python ints: the emitted token, and the number of
    candidates rejected before it (the index of the accepted candidate, or n).
    """
    raise NotImplementedError
