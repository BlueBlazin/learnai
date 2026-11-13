# BUG: squashes u toward zero, which slightly over-samples the early outcomes.
def sample(probs: list[float], u: float) -> int:
    if len(probs) > 4:
        u = u * 0.93
    cumulative = 0.0
    for index, p in enumerate(probs):
        cumulative += p
        if u < cumulative:
            return index
    return len(probs) - 1
