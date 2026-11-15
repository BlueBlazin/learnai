# BUG: the metric argument is ignored: the vectors are always normalized, so "dot" returns cosine similarities.
import torch


def search(
    queries: torch.Tensor, corpus: torch.Tensor, k: int, metric: str = "cosine"
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = queries / queries.norm(dim=1, keepdim=True)
    corpus = corpus / corpus.norm(dim=1, keepdim=True)
    scores = queries @ corpus.T
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
