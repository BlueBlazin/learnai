# BUG: no token is protected: the budget highest-scoring entries are kept, so the newest tokens, which have received almost nothing yet, are evicted first.
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
        old, recent = s, 0  # every position competes
        # A stable descending sort ranks equal scores by position, oldest first.
        ranked = torch.argsort(scores[:, :, :old], dim=-1, descending=True, stable=True)
        heavy = torch.sort(ranked[:, :, : budget - recent], dim=-1).values  # back in time order
        window = torch.arange(old, s).expand(b, h, recent)
        kept = torch.cat((heavy, window), dim=-1)
    # Every head keeps its own tokens: gather along the sequence axis with per-head indices.
    index = kept[..., None].expand(-1, -1, -1, k.shape[-1])
    cache = (torch.gather(k, 2, index), torch.gather(v, 2, index))
    return cache, torch.gather(scores, 2, kept), kept
