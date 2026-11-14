from collections.abc import Callable

import mlx.core as mx


def perplexity(
    forward: Callable[[mx.array], mx.array],
    token_ids: mx.array,
    max_len: int,
    stride: int,
) -> mx.array:
    n = token_ids.shape[0]
    total = mx.zeros((), dtype=mx.float32)
    scored = 1  # tokens before this index are done; token 0 has no context to be scored from
    for begin in range(0, n, stride):
        end = min(begin + max_len, n)
        logits = forward(token_ids[begin:end])
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # (end - begin, V)
        # Row t of this window predicts token begin + t + 1. Only the tokens that no
        # earlier window scored are counted; the ones before them are context.
        rows = mx.arange(scored - 1 - begin, end - 1 - begin)
        total = total + logp[rows, token_ids[scored:end]].sum()
        scored = end
        if end == n:
            break
    # Every token but the first was scored exactly once: n - 1 terms. Natural logarithm.
    return mx.exp(-total / (n - 1))
