# BUG: correct values, but computed with Python loops over rows and history tokens, one tensor element at a time.
import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    out = logits.clone()
    for b, row in enumerate(prev_tokens.tolist()):
        for token in set(row):
            if token == pad_id:
                continue
            value = float(logits[b, token])
            out[b, token] = value / penalty if value > 0 else value * penalty
    return out
