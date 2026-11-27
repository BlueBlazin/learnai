# BUG: repeat_kv_heads expands a new axis placed before the group axis and merges the two, so the copies vary slowest and the heads come out tiled (0, 1, 0, 1).
import torch


def repeat_kv_heads(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, G, T, D = k.shape
    return k[:, None].expand(B, n_rep, G, T, D).reshape(B, n_rep * G, T, D)


def expand_for_beams(x: torch.Tensor, K: int) -> torch.Tensor:
    return x.repeat_interleave(K, dim=0)


def tile_positions(pos: torch.Tensor, B: int) -> torch.Tensor:
    return pos[None, :].expand(B, -1)
