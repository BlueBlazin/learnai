import math

import mlx.core as mx


def causal_self_attention(
    x: mx.array,
    w_q: mx.array,
    w_k: mx.array,
    w_v: mx.array,
    w_o: mx.array,
) -> mx.array:
    seq = x.shape[1]
    # Weights are (out, in), so a projection is x @ W^T: (B, T, d) -> (B, T, out).
    q, k, v = x @ w_q.T, x @ w_k.T, x @ w_v.T
    # scores[b, i, j] = q[b, i] . k[b, j] / sqrt(d_k): row i is the query, column j the key.
    scores = q @ mx.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])
    # Position i may look at j <= i. Mask the logits (not the probabilities) so that
    # the softmax renormalizes over the visible positions only.
    future = mx.triu(mx.ones((seq, seq)), k=1) > 0
    scores = mx.where(future, float("-inf"), scores)
    attn = mx.softmax(scores, axis=-1)
    return (attn @ v) @ w_o.T
