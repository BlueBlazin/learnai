import mlx.core as mx


def typical_accept(
    p: mx.array, draft_tokens: mx.array, epsilon: float, delta: float
) -> tuple[int, int]:
    k = draft_tokens.shape[0]
    # Entropy of every row in nats. A zero probability contributes 0, not 0 * log(0):
    # replace it by 1 inside the logarithm, where p * log(1) is exactly 0.
    entropy = -mx.sum(p * mx.log(mx.where(p > 0, p, 1.0)), axis=-1)
    # The bar is epsilon for a confident target and drops with exp(-H) when it is unsure.
    threshold = mx.minimum(delta * mx.exp(-entropy), epsilon)
    # Row i is the position of draft_tokens[i]: the target's probability of each draft token.
    prob = p[mx.arange(k), draft_tokens]
    accepted = (prob > threshold[:k]).astype(mx.int32)
    # A running product is 1 up to the first failure and 0 from there on.
    n_accepted = int(mx.sum(mx.cumprod(accepted, axis=0)).item())
    # The target's own greedy token follows the accepted prefix, whatever the draft said.
    return n_accepted, int(mx.argmax(p[n_accepted]).item())
