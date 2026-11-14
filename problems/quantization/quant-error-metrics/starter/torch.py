import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error: the mean of (a - b)**2 over every element.

    a, b: float32 tensors of the same shape, 1 to 4 dimensions.
    Returns a 0-d float32 tensor.
    """
    raise NotImplementedError


def sqnr_db(ref: torch.Tensor, approx: torch.Tensor) -> torch.Tensor:
    """Signal-to-quantization-noise ratio in decibels: 10 log10(sum ref**2 / sum (ref - approx)**2).

    ref, approx: float32 tensors of the same shape, 1 to 4 dimensions; ref is not all zeros.
    Returns a 0-d float32 tensor; +inf when approx equals ref everywhere.
    """
    raise NotImplementedError


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Largest absolute difference: the maximum of |a - b| over every element.

    a, b: float32 tensors of the same shape, 1 to 4 dimensions.
    Returns a 0-d float32 tensor.
    """
    raise NotImplementedError


def output_error(w: torch.Tensor, w_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Relative error of a linear layer's output: ||x w^T - x w_hat^T||_F / ||x w^T||_F.

    w, w_hat: float32 tensors of shape (out, in). x: float32 tensor of shape (n, in).
    Returns a 0-d float32 tensor.
    """
    raise NotImplementedError
