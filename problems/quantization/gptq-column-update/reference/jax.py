import jax
import jax.numpy as jnp


def inverse_hessian_factor(x_calib: jax.Array, percdamp: float) -> jax.Array:
    n, in_dim = x_calib.shape
    # The Hessian of the layer's squared output error with respect to one weight row.
    h = (2.0 / n) * (x_calib.T @ x_calib)
    # Dampening: add a fraction of the average diagonal, so that H is safely invertible
    # even when the calibration inputs do not span every direction.
    h = h + percdamp * jnp.mean(jnp.diagonal(h)) * jnp.eye(in_dim)
    # H^-1 through a Cholesky factorization of H, then the factor of H^-1 itself:
    # the upper-triangular U with H^-1 = U^T U.
    lower = jnp.linalg.cholesky(h)
    lower_inv = jax.scipy.linalg.solve_triangular(lower, jnp.eye(in_dim), lower=True)
    h_inv = lower_inv.T @ lower_inv
    return jnp.linalg.cholesky(h_inv, upper=True)


def gptq_quantize(w: jax.Array, u: jax.Array, bits: int) -> tuple[jax.Array, jax.Array]:
    qmax = 2 ** (bits - 1) - 1
    # The grid is fixed up front from the original weights: one symmetric scale per row.
    max_abs = jnp.max(jnp.abs(w), axis=1)
    scale = jnp.where(max_abs > 0, max_abs / qmax, jnp.ones_like(max_abs))
    codes = []
    for j in range(w.shape[1]):
        # Quantize column j as it is now, after the corrections from columns 0..j-1.
        q = jnp.clip(jnp.round(w[:, j] / scale), -qmax, qmax)
        codes.append(q)
        # The error of this column, in units of the factor's diagonal entry ...
        err = (w[:, j] - q * scale) / u[j, j]
        # ... is pushed into the columns not yet quantized, along row j of the factor.
        # The row is zero before column j, so earlier columns are left alone.
        w = w - jnp.outer(err, u[j])
    return jnp.stack(codes, axis=1).astype(jnp.int8), scale
