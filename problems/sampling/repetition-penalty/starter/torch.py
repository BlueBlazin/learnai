import torch


def apply_repetition_penalty(
    logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float, pad_id: int
) -> torch.Tensor:
    """Penalize the logits of tokens that already occur in each row's history.

    logits: float32 tensor of shape (B, V).
    prev_tokens: integer tensor of shape (B, T), token ids in [0, V); entries equal to
        pad_id are padding.
    penalty: the repetition penalty, >= 1.
    pad_id: the padding id, in [0, V); never penalized.
    Returns a new float32 tensor of shape (B, V). The inputs are not modified.
    """
    raise NotImplementedError
