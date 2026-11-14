# BUG: always takes the next token from the last row of logits (the `logits[-1]` habit), which is only right after a full match; after a rejection that row was computed from rejected tokens.
import torch


def verify_greedy(draft_tokens: torch.Tensor, target_logits: torch.Tensor) -> tuple[int, int]:
    greedy = torch.argmax(target_logits, dim=-1)
    match = (greedy[:-1] == draft_tokens).to(torch.int64)
    n_accepted = int(torch.cumprod(match, dim=0).sum())
    return n_accepted, int(greedy[-1])
