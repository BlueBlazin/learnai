import jax
import jax.numpy as jnp


def multi_draft_step(
    p: jax.Array,
    q: jax.Array,
    candidates: jax.Array,
    u_accept: jax.Array,
    u_resample: float,
) -> tuple[int, int]:
    residual = p  # the distribution the next candidate has to deliver; sums to 1
    n = candidates.shape[0]
    for i in range(n):
        x = candidates[i]
        # Accept with probability min(1, residual[x] / q[x]), without the division.
        if bool(u_accept[i] * q[x] < residual[x]):
            return int(x), i
        # Rejected: what acceptance could not deliver is max(0, residual - q). Normalized,
        # it is the target of the next candidate, and of the final sample.
        residual = jnp.maximum(residual - q, 0.0)
        residual = residual / jnp.sum(residual)
    # Every candidate was rejected: sample the last residual by inverse CDF.
    cdf = jnp.cumsum(residual)
    return int(jnp.sum(cdf <= u_resample * cdf[-1])), n
