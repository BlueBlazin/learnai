# BUG: among equal scores the newer token is kept (the ranking is done on the reversed sequence), not the older one.
import torch

Cache = tuple[torch.Tensor, torch.Tensor]  # (k, v)


def h2o_step(
    cache: Cache,
    scores: torch.Tensor | None,
    attn_weights: torch.Tensor,
    budget: int,
    recent: int,
) -> tuple[Cache, torch.Tensor, torch.Tensor]:
    k, v = cache
    b, h, t, s = attn_weights.shape
    # What each cached token received in this step: the column sums over the new queries.
    received = attn_weights.sum(dim=2)
    if scores is None:
        scores = received
    else:
        # The T new tokens start at 0; the older ones keep what they have accumulated.
        scores = torch.cat((scores, torch.zeros(b, h, t)), dim=2) + received
    if s <= budget:
        kept = torch.arange(s).expand(b, h, s)
    else:
        old = s - recent  # positions 0 .. old - 1 compete; the last `recent` are safe
        # A stable descending sort ranks equal scores by position, oldest first.
        flipped = torch.flip(scores[:, :, :old], dims=(-1,))
        ranked = old - 1 - torch.argsort(flipped, dim=-1, descending=True, stable=True)
        heavy = torch.sort(ranked[:, :, : budget - recent], dim=-1).values  # back in time order
        window = torch.arange(old, s).expand(b, h, recent)
        kept = torch.cat((heavy, window), dim=-1)
    # Every head keeps its own tokens: gather along the sequence axis with per-head indices.
    index = kept[..., None].expand(-1, -1, -1, k.shape[-1])
    cache = (torch.gather(k, 2, index), torch.gather(v, 2, index))
    return cache, torch.gather(scores, 2, kept), kept
