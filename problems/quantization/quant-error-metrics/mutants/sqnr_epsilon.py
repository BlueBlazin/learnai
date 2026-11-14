# BUG: adds 1e-12 to the noise power to avoid dividing by zero, so an exact approximation reports a finite SQNR instead of +inf.
import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean()


def sqnr_db(ref: torch.Tensor, approx: torch.Tensor) -> torch.Tensor:
    signal = (ref**2).sum()
    noise = ((ref - approx) ** 2).sum()
    return 10.0 * torch.log10(signal / (noise + 1e-12))


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().max()


def output_error(w: torch.Tensor, w_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = x @ w.T
    y_hat = x @ w_hat.T
    return torch.sqrt(((y - y_hat) ** 2).sum()) / torch.sqrt((y**2).sum())
