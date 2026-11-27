# BUG: tile_positions adds the new axis after the position axis and returns (T, B), with out[t, b] = pos[t]; the shape is right when B == T.
import torch


def repeat_kv_heads(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    return k.repeat_interleave(n_rep, dim=1)


def expand_for_beams(x: torch.Tensor, K: int) -> torch.Tensor:
    return x.repeat_interleave(K, dim=0)


def tile_positions(pos: torch.Tensor, B: int) -> torch.Tensor:
    return pos[:, None].expand(-1, B)
