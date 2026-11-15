# BUG: rebuilds H^-1 = U^T U and uses its rows for every column, without removing the already quantized columns from it; only the Cholesky rows account for that elimination.
import torch


def inverse_hessian_factor(x_calib: torch.Tensor, percdamp: float) -> torch.Tensor:
    n, in_dim = x_calib.shape
    h = (2.0 / n) * (x_calib.T @ x_calib)
    h = h + percdamp * torch.diagonal(h).mean() * torch.eye(in_dim)
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(h_inv, upper=True)


def gptq_quantize(w: torch.Tensor, u: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    max_abs = w.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    h_inv = u.T @ u
    codes = []
    for j in range(w.shape[1]):
        q = torch.clamp(torch.round(w[:, j] / scale), -qmax, qmax)
        codes.append(q)
        err = (w[:, j] - q * scale) / h_inv[j, j]
        w = w - torch.outer(err, torch.cat([torch.zeros(j), h_inv[j, j:]]))
    return torch.stack(codes, dim=1).to(torch.int8), scale
