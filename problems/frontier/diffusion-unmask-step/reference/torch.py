import torch


def schedule(n_masked: int, steps: int) -> list[int]:
    # Spread the masked positions evenly; the first n_masked % steps steps take one
    # more, so the counts add up to n_masked and nothing is left for after the end.
    base, extra = divmod(n_masked, steps)
    return [base + 1 if i < extra else base for i in range(steps)]


def unmask_step(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    mask_id: int,
    n_unmask: int,
    temperature: float,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    masked = tokens == mask_id
    # The model never predicts the mask token itself.
    logits = torch.where(torch.arange(logits.shape[-1]) == mask_id, float("-inf"), logits)

    # A candidate token for every position, from that position's own uniform number.
    if temperature == 0:
        candidates = torch.argmax(logits, dim=-1)
    else:
        cdf = torch.cumsum(torch.softmax(logits / temperature, dim=-1), dim=-1)
        # Inverse CDF: the first index whose cumulative sum exceeds u times the total.
        candidates = (cdf <= uniforms[:, None] * cdf[:, -1:]).sum(dim=-1)

    # Confidence: the probability the model itself (temperature 1) gives the candidate
    # that was drawn, not the probability of its favorite token.
    probs = torch.softmax(logits, dim=-1)
    confidence = torch.take_along_dim(probs, candidates[:, None], dim=-1)[:, 0]
    # A position that is already unmasked can never be chosen.
    confidence = torch.where(masked, confidence, float("-inf"))

    # Commit the n most confident masked positions. A stable descending sort puts
    # equal confidences in index order; rank[i] is the place of position i in it.
    n = min(n_unmask, int(masked.sum()))
    order = torch.argsort(confidence, descending=True, stable=True)
    rank = torch.argsort(order)
    return torch.where(rank < n, candidates, tokens)
