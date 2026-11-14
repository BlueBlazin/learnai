import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # One number for the whole tensor: the mean runs over every element.
    return ((a - b) ** 2).mean()


def sqnr_db(ref: torch.Tensor, approx: torch.Tensor) -> torch.Tensor:
    signal = (ref**2).sum()
    noise = ((ref - approx) ** 2).sum()
    # A ratio of powers (squared quantities) takes 10 log10. Zero noise gives +inf.
    return 10.0 * torch.log10(signal / noise)


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().max()


def output_error(w: torch.Tensor, w_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # A linear layer computes x w^T: one row of outputs per row of x.
    y = x @ w.T
    y_hat = x @ w_hat.T
    # Frobenius norms over the whole output, normalized by the exact output.
    return torch.sqrt(((y - y_hat) ** 2).sum()) / torch.sqrt((y**2).sum())
