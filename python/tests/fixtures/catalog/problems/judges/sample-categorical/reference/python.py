def sample(probs: list[float], u: float) -> int:
    """Inverse-CDF sampling: the first index whose cumulative probability exceeds u."""
    cumulative = 0.0
    for index, p in enumerate(probs):
        cumulative += p
        if u < cumulative:
            return index
    return len(probs) - 1  # guard against cumulative rounding just below 1
