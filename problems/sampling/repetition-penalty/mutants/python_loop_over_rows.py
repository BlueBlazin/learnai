# BUG: correct values, but computed with a Python loop over the batch rows, which is far slower than one batched scatter.
import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    out = logits.clone()
    for b in range(logits.shape[0]):
        tokens = prev_tokens[b]
        tokens = tokens[tokens != pad_id]
        seen = logits[b, tokens]
        out[b, tokens] = torch.where(seen > 0, seen / penalty, seen * penalty)
    return out
