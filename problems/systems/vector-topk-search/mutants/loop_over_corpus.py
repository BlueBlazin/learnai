# BUG: the scores are computed one corpus vector at a time in a Python loop (N small products instead of one matrix product); the result is right and far too slow.
import torch


def search(
    queries: torch.Tensor, corpus: torch.Tensor, k: int, metric: str = "cosine"
) -> tuple[torch.Tensor, torch.Tensor]:
    if metric == "cosine":
        queries = queries / queries.norm(dim=1, keepdim=True)
    scores = torch.empty(queries.shape[0], corpus.shape[0])
    for j in range(corpus.shape[0]):
        vector = corpus[j]
        if metric == "cosine":
            vector = vector / vector.norm()
        scores[:, j] = queries @ vector
    k = min(k, corpus.shape[0])

    kth = torch.topk(scores, k, dim=1).values[:, -1:]
    above = scores > kth
    tied = scores == kth
    room = k - above.sum(dim=1, keepdim=True)
    keep = above | (tied & (tied.cumsum(dim=1) <= room))
    index = keep.nonzero()[:, 1].view(-1, k)

    picked = scores.gather(1, index)
    order = torch.argsort(picked, dim=1, descending=True, stable=True)
    return index.gather(1, order), picked.gather(1, order)
