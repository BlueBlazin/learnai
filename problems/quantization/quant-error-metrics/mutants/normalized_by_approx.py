# BUG: output_error divides by the norm of the approximate output x w_hat^T instead of the exact output x w^T.
import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean()


def sqnr_db(ref: torch.Tensor, approx: torch.Tensor) -> torch.Tensor:
    signal = (ref**2).sum()
    noise = ((ref - approx) ** 2).sum()
    return 10.0 * torch.log10(signal / noise)


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().max()


def output_error(w: torch.Tensor, w_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = x @ w.T
    y_hat = x @ w_hat.T
    return torch.sqrt(((y - y_hat) ** 2).sum()) / torch.sqrt((y_hat**2).sum())
