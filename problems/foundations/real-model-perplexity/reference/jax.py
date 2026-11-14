from collections.abc import Callable

import jax
import jax.numpy as jnp


def perplexity(
    forward: Callable[[jax.Array], jax.Array],
    token_ids: jax.Array,
    max_len: int,
    stride: int,
) -> jax.Array:
    n = token_ids.shape[0]
    total = jnp.zeros((), dtype=jnp.float32)
    scored = 1  # tokens before this index are done; token 0 has no context to be scored from
    for begin in range(0, n, stride):
        end = min(begin + max_len, n)
        logp = jax.nn.log_softmax(forward(token_ids[begin:end]), axis=-1)  # (end - begin, V)
        # Row t of this window predicts token begin + t + 1. Only the tokens that no
        # earlier window scored are counted; the ones before them are context.
        rows = jnp.arange(scored - 1 - begin, end - 1 - begin)
        total = total + logp[rows, token_ids[scored:end]].sum()
        scored = end
        if end == n:
            break
    # Every token but the first was scored exactly once: n - 1 terms. Natural logarithm.
    return jnp.exp(-total / (n - 1))
