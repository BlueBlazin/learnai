import jax
import jax.numpy as jnp


def gdn_gates(
    a: jax.Array, b: jax.Array, A_log: jax.Array, dt_bias: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # softplus is positive and exp(A_log) is positive, so the exponent is negative
    # and g lies in (0, 1): the memory can fade, never grow.
    g = jnp.exp(-jnp.exp(A_log) * jax.nn.softplus(a + dt_bias))
    beta = jax.nn.sigmoid(b)
    return g, beta


def gdn_step(
    S: jax.Array, q: jax.Array, k: jax.Array, v: jax.Array, g: jax.Array, beta: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # 1. Decay: every stored association fades by g, one factor per head.
    S = S * g[:, None, None]
    # 2. Write. S k is what the decayed memory returns for this key; only the
    #    difference to v is stored, along the direction k.
    predicted = jnp.einsum("hvk,hk->hv", S, k)
    delta = beta[:, None] * (v - predicted)
    S = S + delta[:, :, None] * k[:, None, :]  # outer product, (H, Dv, Dk)
    # 3. Read from the updated memory: the token sees its own write.
    y = jnp.einsum("hvk,hk->hv", S, q)
    return y, S
