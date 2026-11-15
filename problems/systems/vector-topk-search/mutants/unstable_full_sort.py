# BUG: every row is sorted in full with an unstable sort (the default of torch.sort), so equal scores come out in no particular index order.
import torch


def search(
    queries: torch.Tensor, corpus: torch.Tensor, k: int, metric: str = "cosine"
) -> tuple[torch.Tensor, torch.Tensor]:
    if metric == "cosine":
        queries = queries / queries.norm(dim=1, keepdim=True)
        corpus = corpus / corpus.norm(dim=1, keepdim=True)
    scores = queries @ corpus.T
    k = min(k, corpus.shape[0])
    values, index = torch.sort(scores, dim=1, descending=True)
    return index[:, :k], values[:, :k]
