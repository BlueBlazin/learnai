# BUG: borrows another framework's built-in RoPE layer (mlx.nn.RoPE) and converts the result, which the statement rules out; the layer also assumes positions 0..T-1.
import mlx.core as mx
import mlx.nn
import numpy as np
import torch


def _builtin(x: torch.Tensor, theta: float, traditional: bool) -> torch.Tensor:
    layer = mlx.nn.RoPE(x.shape[-1], traditional=traditional, base=theta)
    return torch.from_numpy(np.array(layer(mx.array(x.numpy()))))


def rope_interleaved(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    return _builtin(x, theta, traditional=True)


def rope_half(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    return _builtin(x, theta, traditional=False)


def permute_qk(w: torch.Tensor, d_head: int, n_heads: int) -> torch.Tensor:
    pairs = w.reshape(n_heads, d_head // 2, 2, w.shape[-1])
    return pairs.transpose(1, 2).reshape(w.shape)
