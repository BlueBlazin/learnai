import mlx.core as mx


def typical_accept(
    p: mx.array, draft_tokens: mx.array, epsilon: float, delta: float
) -> tuple[int, int]:
    """Verify a draft by typical acceptance.

    p: float32, shape (K + 1, V); row i is the target's next-token distribution at the
       position of draft_tokens[i], and row K the one after the last draft token.
       Rows sum to 1 and may contain exact zeros.
    draft_tokens: integer tensor, shape (K,).
    epsilon, delta: floats in (0, 1], the hard and the entropy-dependent threshold.
    Returns (n_accepted, next_token) as two Python ints.
    """
    raise NotImplementedError
