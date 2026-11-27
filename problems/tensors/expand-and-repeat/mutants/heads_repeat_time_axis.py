# BUG: repeat_kv_heads repeats along axis 2, the positions, returning (B, G, T * n_rep, D).
import torch


def repeat_kv_heads(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    return k.repeat_interleave(n_rep, dim=2)


def expand_for_beams(x: torch.Tensor, K: int) -> torch.Tensor:
    return x.repeat_interleave(K, dim=0)


def tile_positions(pos: torch.Tensor, B: int) -> torch.Tensor:
    return pos[None, :].expand(B, -1)
