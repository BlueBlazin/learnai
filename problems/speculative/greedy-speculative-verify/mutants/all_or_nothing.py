# BUG: accepts the draft only if every token matches and otherwise throws the whole block away, discarding the matching prefix before the first mismatch.
import torch


def verify_greedy(draft_tokens: torch.Tensor, target_logits: torch.Tensor) -> tuple[int, int]:
    greedy = torch.argmax(target_logits, dim=-1)
    n_accepted = len(draft_tokens) if bool((greedy[:-1] == draft_tokens).all()) else 0
    return n_accepted, int(greedy[n_accepted])
