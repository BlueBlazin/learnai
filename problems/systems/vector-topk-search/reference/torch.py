import torch


def search(
    queries: torch.Tensor, corpus: torch.Tensor, k: int, metric: str = "cosine"
) -> tuple[torch.Tensor, torch.Tensor]:
    if metric == "cosine":
        # Cosine similarity is the inner product of unit vectors: normalize both sides.
        queries = queries / queries.norm(dim=1, keepdim=True)
        corpus = corpus / corpus.norm(dim=1, keepdim=True)
    scores = queries @ corpus.T  # (Q, N): every query against every vector, one product
    k = min(k, corpus.shape[0])

    # topk finds the k best *values* of a row without sorting it, but which of
    # several equal scores it returns is unspecified. So take only the k-th best value
    # from it, and select by value: everything above it, then the tied entries in
    # index order until the row has k.
    kth = torch.topk(scores, k, dim=1).values[:, -1:]
    above = scores > kth
    tied = scores == kth
    room = k - above.sum(dim=1, keepdim=True)
    keep = above | (tied & (tied.cumsum(dim=1) <= room))
    index = keep.nonzero()[:, 1].view(-1, k)  # exactly k per row, in ascending index

    # Only k scores per row are left to sort. A stable sort keeps the lower index
    # first among equal scores.
    picked = scores.gather(1, index)
    order = torch.argsort(picked, dim=1, descending=True, stable=True)
    return index.gather(1, order), picked.gather(1, order)
