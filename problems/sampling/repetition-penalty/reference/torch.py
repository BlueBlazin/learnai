import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    # seen[b, v]: does token v occur in row b's history? One scatter along the vocab
    # axis marks every history entry of every row. A token that occurs several times is
    # written several times with the same value, so it is still penalized once, and the
    # value written for a padding entry is False, so pad_id is never marked.
    seen = torch.zeros_like(logits, dtype=torch.bool).scatter(1, prev_tokens, prev_tokens != pad_id)
    # Push the logit toward "less likely" whatever its sign.
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalized, logits)
