# BUG: tile_positions repeats each position B times, which is the (T, B) layout flattened, and reshapes that to (B, T): row 0 starts 0, 0, ... instead of 0, 1, ...
import torch


def repeat_kv_heads(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    return k.repeat_interleave(n_rep, dim=1)


def expand_for_beams(x: torch.Tensor, K: int) -> torch.Tensor:
    return x.repeat_interleave(K, dim=0)


def tile_positions(pos: torch.Tensor, B: int) -> torch.Tensor:
    return pos.repeat_interleave(B).reshape(B, pos.shape[0])
