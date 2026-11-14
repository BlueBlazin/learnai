import torch


def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    t = temperature[:, None]  # (B, 1): one temperature per row, broadcast over the vocabulary
    greedy = t == 0
    # Greedy rows are divided by 1 instead of 0, so no inf or NaN is ever computed, not
    # even in a branch that is thrown away below.
    scaled = logits / torch.where(greedy, 1.0, t)
    # Greedy rows: the first maximum (argmax returns the lowest index among ties) keeps
    # logit 0 and every other entry is masked. Softmax of that row is exactly one-hot.
    top = torch.argmax(logits, dim=-1, keepdim=True)
    is_top = torch.arange(logits.shape[-1]) == top
    one_hot = torch.where(is_top, 0.0, float("-inf"))
    return torch.where(greedy, one_hot, scaled)
