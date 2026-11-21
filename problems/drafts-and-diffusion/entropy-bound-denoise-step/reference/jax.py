import jax
import jax.numpy as jnp


def step_temperature(n: int, n_max: int, t_min: float, t_max: float) -> float:
    # n counts the steps that remain: the first step (n = n_max) is the hottest.
    return t_min + (t_max - t_min) * (n / n_max)


def denoise_step(
    current: jax.Array,
    logits: jax.Array,
    temperature: float,
    entropy_bound: float,
    u_sample: jax.Array,
    u_noise: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    vocab = logits.shape[-1]
    log_p = jax.nn.log_softmax(logits / temperature, axis=-1)
    p = jnp.exp(log_p)

    # One candidate per position by inverse CDF: the first index whose cumulative
    # probability exceeds the uniform scaled by the row's total.
    cdf = jnp.cumsum(p, axis=-1)
    sampled = (cdf <= u_sample[:, None] * cdf[:, -1:]).sum(axis=-1)

    # Entropy in nats of the distribution that was sampled from. log_p is finite, so a
    # probability that underflowed to 0 contributes 0 * log_p = 0.
    entropy = -(p * log_p).sum(axis=-1)

    # Rank ascending (stable: ties keep index order). A position is accepted while the
    # entropies ranked strictly before it sum to at most the bound.
    order = jnp.argsort(entropy, stable=True)
    ranked = entropy[order]
    before = jnp.cumsum(ranked) - ranked
    accepted = (before <= entropy_bound)[jnp.argsort(order)]  # back to position order

    # Every other position is renoised. `current` survives nowhere.
    noise = jnp.floor(u_noise * vocab).astype(jnp.int32)
    canvas = jnp.where(accepted, sampled, noise)
    return canvas, accepted, jnp.argmax(logits, axis=-1)
