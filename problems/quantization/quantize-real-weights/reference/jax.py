import math

import jax
import jax.numpy as jnp

FLOAT_BITS = 16  # scales, biases, and unquantized tensors are stored as 16-bit floats


def quantize_dequantize(w: jax.Array, bits: int, group_size: int) -> jax.Array:
    out_dim, in_dim = w.shape
    levels = 2**bits - 1  # the largest code
    # Every row is cut into consecutive runs of group_size inputs.
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.min(axis=-1, keepdims=True)
    hi = groups.max(axis=-1, keepdims=True)
    # The grid starts at the end of the range that is further from zero (code 0) and
    # steps toward the other end: upward from the minimum, or downward from the maximum.
    from_low = jnp.abs(lo) > jnp.abs(hi)
    # JAX divides by a scalar by multiplying with its reciprocal, which is one bit off
    # for a third of all values, and one bit decides a tie in q0 below. Dividing by an
    # array is exact, as in MLX.
    step = (hi - lo) / jnp.full_like(hi, levels)
    step = jnp.maximum(step, 1e-7)  # the floor keeps a constant group finite
    scale = jnp.where(from_low, step, -step)
    edge = jnp.where(from_low, lo, hi)
    # Zero lies -edge / scale steps from the edge. Rounding that to an integer and
    # recomputing the scale puts a grid point exactly on zero.
    q0 = jnp.round(edge / scale)
    scale = jnp.where(q0 != 0, edge / jnp.where(q0 != 0, q0, 1.0), scale)
    bias = jnp.where(q0 != 0, edge, 0.0)
    codes = jnp.clip(jnp.round((groups - bias) / scale), 0, levels)
    return (codes * scale + bias).reshape(out_dim, in_dim)


def quantize_report(weights: dict[str, jax.Array], bits: int, group_size: int) -> dict:
    sqnr_db = {}
    signal_total = noise_total = 0.0
    stored_bits = parameters = 0
    for name, w in weights.items():
        parameters += w.size
        if w.ndim != 2:
            stored_bits += FLOAT_BITS * w.size  # norm gains stay as they are
            continue
        w_hat = quantize_dequantize(w, bits, group_size)
        # Row sums first, then their sum: a single float32 sum over 10**8 squares
        # would lose digits. The totals are accumulated as Python floats (float64).
        signal = float((w * w).sum(axis=-1).sum())
        noise = float(((w - w_hat) ** 2).sum(axis=-1).sum())
        sqnr_db[name] = 10.0 * math.log10(signal / noise)
        signal_total += signal
        noise_total += noise
        # The codes, plus one scale and one bias per group. Python ints: no overflow.
        stored_bits += bits * w.size + 2 * FLOAT_BITS * (w.size // group_size)
    return {
        "sqnr_db": sqnr_db,
        # Pooled over the model: total signal power over total noise power, which
        # weights every layer by its size. Not the mean of the per-layer decibels.
        "model_sqnr_db": 10.0 * math.log10(signal_total / noise_total),
        "bits_per_weight": stored_bits / parameters,
        "worst": sorted(sqnr_db, key=sqnr_db.get)[:3],  # lowest SQNR first
    }
