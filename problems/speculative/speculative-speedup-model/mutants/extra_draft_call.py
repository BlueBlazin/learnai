# BUG: charges k + 1 draft calls per round, as if the draft model also had to produce the bonus token.
def expected_tokens(alpha: float, k: int) -> float:
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    return expected_tokens(alpha, k) / ((k + 1) * c + 1.0)


def best_k(alpha: float, c: float, k_max: int) -> int:
    return max(range(k_max + 1), key=lambda k: speedup(alpha, k, c))
