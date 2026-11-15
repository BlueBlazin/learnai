import torch


def search(
    queries: torch.Tensor, corpus: torch.Tensor, k: int, metric: str = "cosine"
) -> tuple[torch.Tensor, torch.Tensor]:
    """The k most similar corpus vectors of every query, best first.

    queries: float32, shape (Q, D). corpus: float32, shape (N, D). No vector is zero.
    k: how many neighbors to return per query (k >= 1); at most N are returned.
    metric: "cosine" (cosine similarity) or "dot" (inner product).
    Returns (indices, scores): an integer tensor and a float32 tensor, both of shape
    (Q, min(k, N)). Row q lists the corpus indices with the highest scores for query
    q in descending order of score; equal scores are ordered by index, lower first.
    """
    raise NotImplementedError
