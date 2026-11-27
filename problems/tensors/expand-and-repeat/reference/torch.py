import torch


def repeat_kv_heads(k: torch.Tensor, n_rep: int) -> torch.Tensor:
    # repeat_interleave repeats each entry along the axis in place: groups 0, 1 become
    # heads 0, 0, 1, 1, so head g * n_rep + r is group g.
    return k.repeat_interleave(n_rep, dim=1)


def expand_for_beams(x: torch.Tensor, K: int) -> torch.Tensor:
    # The same along the batch axis: rows 0, 1 become rows 0, 0, 1, 1.
    return x.repeat_interleave(K, dim=0)


def tile_positions(pos: torch.Tensor, B: int) -> torch.Tensor:
    # A new leading axis of size 1, expanded to B. expand copies nothing: every row
    # of the result is the memory of pos.
    return pos[None, :].expand(B, -1)
