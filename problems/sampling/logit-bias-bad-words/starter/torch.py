import torch


def apply_bias(
    logits: torch.Tensor,
    bias: dict[int, float],
    allowed_token_ids: list[int] | None,
    bad_words: list[list[int]],
    history: list[int],
) -> torch.Tensor:
    """Apply a logit bias, an allow-list, and banned sequences to one row of logits.

    logits: float32 tensor of shape (V,); entries may be -inf.
    bias: token id -> value added to that token's logit.
    allowed_token_ids: None, or the token ids that may be sampled; all others are removed.
    bad_words: banned token sequences; the last token of a sequence is removed when
        `history` ends with the tokens before it (always, for a one-token sequence).
    history: token ids of the sequence so far, oldest first.
    Returns a float32 tensor of shape (V,): -inf for removed tokens (whatever their
    bias), logits[v] + bias[v] for the others.
    """
    raise NotImplementedError
