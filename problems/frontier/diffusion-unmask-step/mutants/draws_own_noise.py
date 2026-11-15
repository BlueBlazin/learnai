# BUG: ignores the supplied uniforms and draws its own with torch.rand, so the result is not reproducible and cannot be checked.
import torch


def schedule(n_masked: int, steps: int) -> list[int]:
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
    logits = torch.where(torch.arange(logits.shape[-1]) == mask_id, float("-inf"), logits)

    if temperature == 0:
        candidates = torch.argmax(logits, dim=-1)
    else:
        cdf = torch.cumsum(torch.softmax(logits / temperature, dim=-1), dim=-1)
        candidates = (cdf <= torch.rand(cdf.shape[0])[:, None] * cdf[:, -1:]).sum(dim=-1)

    probs = torch.softmax(logits, dim=-1)
    confidence = torch.take_along_dim(probs, candidates[:, None], dim=-1)[:, 0]
    confidence = torch.where(masked, confidence, float("-inf"))

    n = min(n_unmask, int(masked.sum()))
    order = torch.argsort(confidence, descending=True, stable=True)
    rank = torch.argsort(order)
    return torch.where(rank < n, candidates, tokens)
