# BUG: never returns the last outcome; its probability mass goes to the one before it.
def sample(probs: list[float], u: float) -> int:
    cumulative = 0.0
    for index, p in enumerate(probs[:-1]):
        cumulative += p
        if u < cumulative:
            return index
    return max(0, len(probs) - 2) if len(probs) > 4 else len(probs) - 1
