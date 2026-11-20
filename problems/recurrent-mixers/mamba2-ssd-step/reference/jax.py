import jax
import jax.numpy as jnp


def ssd_step(
    h: jax.Array,
    x: jax.Array,
    dt_raw: jax.Array,
    B: jax.Array,
    C: jax.Array,
    A_log: jax.Array,
    D: jax.Array,
    dt_bias: jax.Array,
    dt_limit: tuple[float, float],
) -> tuple[jax.Array, jax.Array]:
    # The step size of each head: positive, then clamped to the configured range.
    dt = jnp.clip(jax.nn.softplus(dt_raw + dt_bias), dt_limit[0], dt_limit[1])
    # Group j serves the heads j * r .. j * r + r - 1: repeat, do not tile.
    reps = h.shape[0] // B.shape[0]
    B = jnp.repeat(B, reps, axis=0)  # (H, N)
    C = jnp.repeat(C, reps, axis=0)
    # h <- h * exp(dt * A) + dt * x B^T, with A = -exp(A_log) < 0.
    decay = jnp.exp(-dt * jnp.exp(A_log))
    h = h * decay[:, None, None] + (dt[:, None] * x)[:, :, None] * B[:, None, :]
    # Read the updated state, plus the skip connection.
    y = jnp.einsum("hpn,hn->hp", h, C) + D[:, None] * x
    return y, h


def ssd_matrix(dt: jax.Array, B: jax.Array, C: jax.Array, A_log: jax.Array) -> jax.Array:
    steps, heads = dt.shape
    reps = heads // B.shape[1]
    B = jnp.repeat(B, reps, axis=1)  # (T, H, N)
    C = jnp.repeat(C, reps, axis=1)
    scores = jnp.einsum("thn,shn->hts", C, B)  # C_t . B_s
    # exp(sum of dt_r * A for s < r <= t) = exp(c_t - c_s), c the running sum of dt * A.
    c = jnp.cumsum(-dt * jnp.exp(A_log), axis=0).T  # (H, T)
    lower = jnp.tril(jnp.ones((steps, steps), dtype=bool))
    decay = jnp.exp(jnp.where(lower, c[:, :, None] - c[:, None, :], -jnp.inf))
    return scores * decay * dt.T[:, None, :]  # the input at s enters scaled by dt_s
