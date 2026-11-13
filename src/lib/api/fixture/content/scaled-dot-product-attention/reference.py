import math

{{import}}


def attention(q: {{T}}, k: {{T}}, v: {{T}}, mask: {{T}} | None = None) -> {{T}}:
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
