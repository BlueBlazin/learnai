# BUG: repeat_kv_heads repeats the whole group axis (repeat is a tile), giving heads 0, 1, 0, 1 where 0, 0, 1, 1 is wanted: head h reads group h % G instead of h // n_rep.
import torch


def repeat_kv_heads(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    return k.repeat(1, n_rep, 1, 1)


def expand_for_beams(x: torch.Tensor, K: int) -> torch.Tensor:
    return x.repeat_interleave(K, dim=0)


def tile_positions(pos: torch.Tensor, B: int) -> torch.Tensor:
    return pos[None, :].expand(B, -1)
