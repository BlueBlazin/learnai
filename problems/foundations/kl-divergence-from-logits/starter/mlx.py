import mlx.core as mx


def kl_from_logits(p_logits: mx.array, q_logits: mx.array) -> mx.array:
    """KL(softmax(p_logits) || softmax(q_logits)) along the last axis, in nats.

    p_logits, q_logits: float32 arrays of the same shape (..., V); entries are finite
        or -inf (a masked token, probability exactly 0). p is the reference distribution.
    A term is 0 where p is 0, and +inf where q is 0 and p is not (p_logit finite).
    Returns a float32 array of shape (...): each value is >= 0 or +inf, never NaN.
    """
    raise NotImplementedError
