{{import}}


def logsumexp(x: {{T}}, dim: int = -1) -> {{T}}:
    """Return log(sum(exp(x))) along `dim`; `dim` is removed from the shape."""
    m = x.max(dim=dim, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    total = torch.exp(x - m).sum(dim=dim)
    return m.squeeze(dim) + torch.log(total)
