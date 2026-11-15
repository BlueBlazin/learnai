import jax
import jax.numpy as jnp


def _quantize_groups(w: jax.Array, bits: int, group_size: int) -> jax.Array:
    # Group-wise affine quantization followed by dequantization (as in groupwise-4bit):
    # every run of group_size inputs in a row gets its own minimum and step.
    out_dim, in_dim = w.shape
    levels = 2**bits - 1
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = jnp.min(groups, axis=-1, keepdims=True)
    hi = jnp.max(groups, axis=-1, keepdims=True)
    scale = jnp.where(hi > lo, (hi - lo) / levels, jnp.ones_like(hi))
    codes = jnp.clip(jnp.round((groups - lo) / scale), 0, levels)
    return (codes * scale + lo).reshape(out_dim, in_dim)


def awq_scales(
    w: jax.Array, x_calib: jax.Array, bits: int, group_size: int, grid: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # How large each input channel is on the calibration data: the mean magnitude.
    act = jnp.mean(jnp.abs(x_calib), axis=0)
    target = x_calib @ w.T  # what the layer computes before quantization
    candidates, losses = [], []
    for alpha in grid:
        # alpha = 0 is no scaling; alpha = 1 scales each channel by its magnitude.
        s = jnp.maximum(act**alpha, 1e-4)
        # Normalize so that the largest and the smallest scale are reciprocal.
        s = s / jnp.sqrt(jnp.max(s) * jnp.min(s))
        # Scale the columns up, quantize, and scale back down. Dividing the weights by
        # s is the same as dividing the activations by s: x @ (q / s).T == (x / s) @ q.T.
        w_hat = _quantize_groups(w * s, bits, group_size) / s
        # The objective is the error of the layer's output, not of its weights.
        losses.append(jnp.mean((target - x_calib @ w_hat.T) ** 2))
        candidates.append(s)
    # argmin returns the first minimum: ties go to the earlier grid entry.
    best = int(jnp.argmin(jnp.stack(losses)))
    return candidates[best], grid[best]
