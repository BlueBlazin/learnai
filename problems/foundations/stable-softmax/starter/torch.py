import torch


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Softmax of `x` along `axis`, correct for any float32 logits.

    x: float32 tensor with 1 to 4 dimensions; entries may be -inf (masked).
    Returns a float32 tensor of the same shape whose slices along `axis` sum to 1.
    """
    raise NotImplementedError


def log_softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Log of the softmax of `x` along `axis`, without underflowing to -inf.

    x: float32 tensor with 1 to 4 dimensions; entries may be -inf (masked).
    Returns a float32 tensor of the same shape.
    """
    raise NotImplementedError
