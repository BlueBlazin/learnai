import jax
import jax.numpy as jnp

NORM_EPS = 1e-6  # rms_norm_eps of DeepSeek-V4; not the `eps` argument


def hc_weights(
    streams: jax.Array,
    fn: jax.Array,
    base: jax.Array,
    scale: jax.Array,
    iters: int,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    steps, n, dim = streams.shape
    # One vector per token: all streams end to end, normalized as a whole, no gain.
    flat = streams.reshape(steps, n * dim)
    flat = flat / jnp.sqrt(jnp.mean(flat * flat, axis=-1, keepdims=True) + NORM_EPS)
    # One linear map gives the raw values of all three weights: N, N, and N * N of them.
    raw = flat @ fn.T
    pre = jax.nn.sigmoid(raw[:, :n] * scale[0] + base[:n]) + eps
    post = 2.0 * jax.nn.sigmoid(raw[:, n : 2 * n] * scale[1] + base[n : 2 * n])
    logits = (raw[:, 2 * n :] * scale[2] + base[2 * n :]).reshape(steps, n, n)
    comb = jax.nn.softmax(logits, axis=-1) + eps

    # Sinkhorn-Knopp: columns once, then (iters - 1) times rows and columns again.
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)

    collapsed = (pre[:, :, None] * streams).sum(axis=1)
    return collapsed, post, comb


def hc_merge(streams: jax.Array, y: jax.Array, post: jax.Array, comb: jax.Array) -> jax.Array:
    # Output stream n = post[n] * y + sum_j comb[j, n] * streams[j]: comb is used transposed.
    return post[:, :, None] * y[:, None, :] + jnp.swapaxes(comb, -1, -2) @ streams
