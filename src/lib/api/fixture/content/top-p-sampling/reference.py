{{import}}


def sample_top_p(logits: {{T}}, p: float, u: {{T}}) -> {{T}}:
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, order = probs.sort(dim=-1, descending=True, stable=True)
    before = sorted_probs.cumsum(dim=-1) - sorted_probs
    sorted_probs = torch.where(before < p, sorted_probs, 0.0)
    cdf = (sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)).cumsum(dim=-1)
    rank = (cdf <= u[:, None]).sum(dim=-1).clamp(max=logits.shape[-1] - 1)
    return order.gather(-1, rank[:, None]).squeeze(-1)
