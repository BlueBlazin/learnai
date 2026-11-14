{{import}}


def causal_bias(length: int, device) -> {{T}}:
    """Return a (length, length) float32 bias on `device`."""
    bias = torch.full((length, length), float("-inf"), device=device)
    return torch.triu(bias, diagonal=1)
