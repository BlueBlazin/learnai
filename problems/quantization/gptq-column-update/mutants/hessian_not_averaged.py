# BUG: builds H as x^T x without the 2 / N factor, so the returned factor is off by a constant.
import torch


def inverse_hessian_factor(x_calib: torch.Tensor, percdamp: float) -> torch.Tensor:
    in_dim = x_calib.shape[1]
    h = x_calib.T @ x_calib
    h = h + percdamp * torch.diagonal(h).mean() * torch.eye(in_dim)
    h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(h_inv, upper=True)


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
