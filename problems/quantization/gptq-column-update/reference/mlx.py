import mlx.core as mx


def inverse_hessian_factor(x_calib: mx.array, percdamp: float) -> mx.array:
    n, in_dim = x_calib.shape
    # The Hessian of the layer's squared output error with respect to one weight row.
    h = (2.0 / n) * (x_calib.T @ x_calib)
    # Dampening: add a fraction of the average diagonal, so that H is safely invertible
    # even when the calibration inputs do not span every direction.
    h = h + percdamp * mx.mean(mx.diagonal(h)) * mx.eye(in_dim)
    # H^-1 through a Cholesky factorization of H, then the factor of H^-1 itself:
    # the upper-triangular U with H^-1 = U^T U. MLX runs factorizations on the CPU only.
    h_inv = mx.linalg.cholesky_inv(mx.linalg.cholesky(h, stream=mx.cpu), stream=mx.cpu)
    return mx.linalg.cholesky(h_inv, upper=True, stream=mx.cpu)


def gptq_quantize(w: mx.array, u: mx.array, bits: int) -> tuple[mx.array, mx.array]:
    qmax = 2 ** (bits - 1) - 1
    # The grid is fixed up front from the original weights: one symmetric scale per row.
    max_abs = mx.max(mx.abs(w), axis=1)
    scale = mx.where(max_abs > 0, max_abs / qmax, mx.ones_like(max_abs))
    codes = []
    for j in range(w.shape[1]):
        # Quantize column j as it is now, after the corrections from columns 0..j-1.
        q = mx.clip(mx.round(w[:, j] / scale), -qmax, qmax)
        codes.append(q)
        # The error of this column, in units of the factor's diagonal entry ...
        err = (w[:, j] - q * scale) / u[j, j]
        # ... is pushed into the columns not yet quantized, along row j of the factor.
        # The row is zero before column j, so earlier columns are left alone.
        w = w - mx.outer(err, u[j])
    return mx.stack(codes, axis=1).astype(mx.int8), scale
