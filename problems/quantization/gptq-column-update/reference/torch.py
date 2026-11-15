import torch


def inverse_hessian_factor(x_calib: torch.Tensor, percdamp: float) -> torch.Tensor:
    n, in_dim = x_calib.shape
    # The Hessian of the layer's squared output error with respect to one weight row.
    h = (2.0 / n) * (x_calib.T @ x_calib)
    # Dampening: add a fraction of the average diagonal, so that H is safely invertible
    # even when the calibration inputs do not span every direction.
    h = h + percdamp * torch.diagonal(h).mean() * torch.eye(in_dim)
    # H^-1 through a Cholesky factorization of H, then the factor of H^-1 itself:
    # the upper-triangular U with H^-1 = U^T U.
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(h_inv, upper=True)


def gptq_quantize(w: torch.Tensor, u: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    # The grid is fixed up front from the original weights: one symmetric scale per row.
    max_abs = w.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    codes = []
    for j in range(w.shape[1]):
        # Quantize column j as it is now, after the corrections from columns 0..j-1.
        q = torch.clamp(torch.round(w[:, j] / scale), -qmax, qmax)
        codes.append(q)
        # The error of this column, in units of the factor's diagonal entry ...
        err = (w[:, j] - q * scale) / u[j, j]
        # ... is pushed into the columns not yet quantized, along row j of the factor.
        # The row is zero before column j, so earlier columns are left alone.
        w = w - torch.outer(err, u[j])
    return torch.stack(codes, dim=1).to(torch.int8), scale
