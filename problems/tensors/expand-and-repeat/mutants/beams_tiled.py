# BUG: expand_for_beams tiles the batch (rows 0, 1, 0, 1), so row b * K + j is row (b * K + j) % B instead of row b.
import torch


def repeat_kv_heads(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    return k.repeat_interleave(n_rep, dim=1)


def expand_for_beams(x: torch.Tensor, K: int) -> torch.Tensor:
    return x.repeat(K, 1)


def tile_positions(pos: torch.Tensor, B: int) -> torch.Tensor:
    return pos[None, :].expand(B, -1)
