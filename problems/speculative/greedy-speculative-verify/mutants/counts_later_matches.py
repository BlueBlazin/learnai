# BUG: counts every position where the draft equals the target's argmax, so matches after the first mismatch are accepted too.
import torch


def verify_greedy(draft_tokens: torch.Tensor, target_logits: torch.Tensor) -> tuple[int, int]:
    greedy = torch.argmax(target_logits, dim=-1)
    match = (greedy[:-1] == draft_tokens).to(torch.int64)
    n_accepted = int(match.sum())
    return n_accepted, int(greedy[n_accepted])
