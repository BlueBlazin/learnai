# BUG: the indices come straight from torch.topk, which does not say which of several equal scores it returns: ties are not broken by the lower index.
import torch


def search(
    queries: torch.Tensor, corpus: torch.Tensor, k: int, metric: str = "cosine"
) -> tuple[torch.Tensor, torch.Tensor]:
    if metric == "cosine":
        queries = queries / queries.norm(dim=1, keepdim=True)
        corpus = corpus / corpus.norm(dim=1, keepdim=True)
    scores = queries @ corpus.T
    k = min(k, corpus.shape[0])
    values, index = torch.topk(scores, k, dim=1)
    return index, values
