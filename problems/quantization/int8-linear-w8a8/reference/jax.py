import jax
import jax.numpy as jnp

QMIN, QMAX = -128, 127


def int8_linear(
    x: jax.Array, w_q: jax.Array, w_scale: jax.Array, bias: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # One affine grid per token (row of x), widened to include 0 so that 0.0 is a code.
    lo = jnp.minimum(jnp.min(x, axis=1, keepdims=True), 0.0)
    hi = jnp.maximum(jnp.max(x, axis=1, keepdims=True), 0.0)
    x_scale = jnp.where(hi > lo, (hi - lo) / (QMAX - QMIN), jnp.ones_like(hi))
    zero_point = jnp.clip(QMIN - jnp.round(lo / x_scale), QMIN, QMAX).astype(jnp.int32)
    x_q = jnp.clip(jnp.round(x / x_scale) + zero_point, QMIN, QMAX).astype(jnp.int8)

    # The integer part of the layer. Widen before multiplying: one product of two int8
    # codes already needs 15 bits, and the sum over `in` of them needs more.
    w32 = w_q.astype(jnp.int32)
    acc = x_q.astype(jnp.int32) @ w32.T  # (n, out), int32

    # x_q stands for x_scale * (x_q - zero_point), so the product owes a correction:
    # sum_j (x_q - z) * w_q = acc - z * sum_j w_q. The row sums of w_q do not depend on x.
    corrected = acc - zero_point * w32.sum(axis=1)

    # Back to real numbers once, at the end: one scale per token times one per output.
    y = corrected.astype(jnp.float32) * (x_scale * w_scale) + bias
    return y, acc
