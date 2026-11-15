import torch


def typical_accept(
    p: torch.Tensor, draft_tokens: torch.Tensor, epsilon: float, delta: float
) -> tuple[int, int]:
    k = draft_tokens.shape[0]
    # Entropy of every row in nats. A zero probability contributes 0, not 0 * log(0):
    # replace it by 1 inside the logarithm, where p * log(1) is exactly 0.
    entropy = -(p * torch.log(torch.where(p > 0, p, torch.ones_like(p)))).sum(dim=-1)
    # The bar is epsilon for a confident target and drops with exp(-H) when it is unsure.
    threshold = torch.clamp(delta * torch.exp(-entropy), max=epsilon)
    # Row i is the position of draft_tokens[i]: the target's probability of each draft token.
    prob = p[torch.arange(k), draft_tokens]
    accepted = (prob > threshold[:k]).to(torch.int64)
    # A running product is 1 up to the first failure and 0 from there on.
    n_accepted = int(torch.cumprod(accepted, dim=0).sum())
    # The target's own greedy token follows the accepted prefix, whatever the draft said.
    return n_accepted, int(torch.argmax(p[n_accepted]))
