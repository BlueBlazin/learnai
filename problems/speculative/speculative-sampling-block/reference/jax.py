import jax
import jax.numpy as jnp


def verify_block(
    p: jax.Array,
    q: jax.Array,
    draft_tokens: jax.Array,
    u_accept: jax.Array,
    u_resample: float,
) -> tuple[int, int]:
    k = q.shape[0]
    rows = jnp.arange(k)
    # Draft i passes its own test when u_i < p_i[x_i] / q_i[x_i]; multiplying through
    # by q avoids the division and the min(1, .).
    passes = u_accept * q[rows, draft_tokens] < p[rows, draft_tokens]
    # Only the prefix before the first failure is accepted: the running product is 1
    # up to the first failed test and 0 afterwards, whatever the later tests say.
    n_accepted = int(jnp.sum(jnp.cumprod(passes.astype(jnp.int32))))
    if n_accepted < k:
        # Rejected at position n: the replacement comes from the residual of that
        # position, max(0, p_n - q_n), normalized.
        weights = jnp.maximum(p[n_accepted] - q[n_accepted], 0.0)
    else:
        # Everything accepted: nothing to correct, so the bonus token is an ordinary
        # sample from the target's distribution in the extra row.
        weights = p[k]
    # Inverse CDF with u_resample. Scaling the uniform by the total normalizes the
    # weights; the first index whose cumulative sum exceeds the threshold equals the
    # number of entries that do not exceed it.
    cdf = jnp.cumsum(weights)
    next_token = int(jnp.sum(cdf <= u_resample * cdf[-1]))
    return n_accepted, next_token
