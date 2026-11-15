import jax
import jax.numpy as jnp


def typical_accept(
    p: jax.Array, draft_tokens: jax.Array, epsilon: float, delta: float
) -> tuple[int, int]:
    k = draft_tokens.shape[0]
    # Entropy of every row in nats. A zero probability contributes 0, not 0 * log(0):
    # replace it by 1 inside the logarithm, where p * log(1) is exactly 0.
    entropy = -jnp.sum(p * jnp.log(jnp.where(p > 0, p, 1.0)), axis=-1)
    # The bar is epsilon for a confident target and drops with exp(-H) when it is unsure.
    threshold = jnp.minimum(delta * jnp.exp(-entropy), epsilon)
    # Row i is the position of draft_tokens[i]: the target's probability of each draft token.
    prob = p[jnp.arange(k), draft_tokens]
    accepted = (prob > threshold[:k]).astype(jnp.int32)
    # A running product is 1 up to the first failure and 0 from there on.
    n_accepted = int(jnp.sum(jnp.cumprod(accepted)))
    # The target's own greedy token follows the accepted prefix, whatever the draft said.
    return n_accepted, int(jnp.argmax(p[n_accepted]))
