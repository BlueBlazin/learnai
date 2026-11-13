{{import}}


def top_k_probs(logits: {{T}}, k: int, temperature: float) -> {{T}}:
    """logits: (B, V). Returns (B, V) probabilities; zero outside the top k."""
    values, indices = (logits / temperature).topk(k, dim=-1)
    return torch.zeros_like(logits).scatter(-1, indices, torch.softmax(values, dim=-1))
