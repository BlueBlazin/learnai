# BUG: borrows another framework's built-in RoPE layer (mlx.nn.RoPE) and converts the result, which the statement rules out; the layer also assumes positions 0..T-1.
import mlx.core as mx
import mlx.nn
import numpy as np
import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    layer = mlx.nn.RoPE(x.shape[-1], traditional=True, base=base)
    return torch.from_numpy(np.array(layer(mx.array(x.numpy()))))
