# BUG: compares draft_tokens[i] with the argmax of row i + 1 instead of row i, as if row 0 belonged to the last verified token.
import torch


def verify_greedy(draft_tokens: torch.Tensor, target_logits: torch.Tensor) -> tuple[int, int]:
    greedy = torch.argmax(target_logits, dim=-1)
    match = (greedy[1:] == draft_tokens).to(torch.int64)
    n_accepted = int(torch.cumprod(match, dim=0).sum())
    return n_accepted, int(greedy[n_accepted])
