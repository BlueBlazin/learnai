# BUG: expand_for_beams ends with a bare squeeze(), which removes the position axis when T == 1 and the row axis when B * K == 1.
import torch


def repeat_kv_heads(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    return k.repeat_interleave(n_rep, dim=1)


def expand_for_beams(x: torch.Tensor, K: int) -> torch.Tensor:
    return x.repeat_interleave(K, dim=0).squeeze()


def tile_positions(pos: torch.Tensor, B: int) -> torch.Tensor:
    return pos[None, :].expand(B, -1)
