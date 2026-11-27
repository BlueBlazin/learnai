import jax
import jax.numpy as jnp


def attention_scores(q: jax.Array, k: jax.Array) -> jax.Array:
    # out[b, h, t, s] = sum over d of q[b, h, t, d] * k[b, h, s, d]. The letters b and
    # h appear in both inputs and in the output: those axes are paired. d appears in
    # both inputs and not in the output: it is multiplied and summed.
    return jnp.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x: jax.Array, w: jax.Array, y: jax.Array) -> jax.Array:
    # out[b] = sum over d and e of x[b, d] * w[d, e] * y[b, e]. The same b on x and y
    # pairs row b of x with row b of y.
    return jnp.einsum("bd,de,be->b", x, w, y)


def per_head_project(x: jax.Array, w: jax.Array) -> jax.Array:
    # out[b, h, t, e] = sum over d of x[b, h, t, d] * w[h, d, e]. The h on w is the h
    # on x, so head h is multiplied by its own matrix.
    return jnp.einsum("bhtd,hde->bhte", x, w)
