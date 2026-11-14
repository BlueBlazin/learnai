# BUG: ignores the extra row K; after a full match it returns the target's token for the last draft position (a repeat of the last draft token) instead of the bonus token.
import torch


def verify_greedy(draft_tokens: torch.Tensor, target_logits: torch.Tensor) -> tuple[int, int]:
    greedy = torch.argmax(target_logits[:-1], dim=-1)
    match = (greedy == draft_tokens).to(torch.int64)
    n_accepted = int(torch.cumprod(match, dim=0).sum())
    return n_accepted, int(greedy[min(n_accepted, len(greedy) - 1)])
