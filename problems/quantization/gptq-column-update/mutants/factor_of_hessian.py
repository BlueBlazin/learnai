# BUG: returns the upper Cholesky factor of H instead of the factor of its inverse.
import torch


def inverse_hessian_factor(x_calib: torch.Tensor, percdamp: float) -> torch.Tensor:
    n, in_dim = x_calib.shape
    h = (2.0 / n) * (x_calib.T @ x_calib)
    h = h + percdamp * torch.diagonal(h).mean() * torch.eye(in_dim)
    return torch.linalg.cholesky(h, upper=True)


def gptq_quantize(w: torch.Tensor, u: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = w.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    codes = []
    for j in range(w.shape[1]):
        q = torch.clamp(torch.round(w[:, j] / scale), -qmax, qmax)
        codes.append(q)
        err = (w[:, j] - q * scale) / u[j, j]
        w = w - torch.outer(err, u[j])
    return torch.stack(codes, dim=1).to(torch.int8), scale
