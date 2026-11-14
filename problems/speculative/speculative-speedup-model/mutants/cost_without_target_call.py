# BUG: the cost of a round is taken to be the k draft calls only (k * c), leaving out the target call that verifies them.
def expected_tokens(alpha: float, k: int) -> float:
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    return expected_tokens(alpha, k) / (k * c)


def best_k(alpha: float, c: float, k_max: int) -> int:
    return max(range(k_max + 1), key=lambda k: speedup(alpha, k, c))
