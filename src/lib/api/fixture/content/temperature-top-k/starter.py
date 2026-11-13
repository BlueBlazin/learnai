{{import}}


def top_k_probs(logits: {{T}}, k: int, temperature: float) -> {{T}}:
    """logits: (B, V). Returns (B, V) probabilities; zero outside the top k."""
    raise NotImplementedError
