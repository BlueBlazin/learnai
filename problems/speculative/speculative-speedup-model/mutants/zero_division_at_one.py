# BUG: uses the closed form for every alpha; at alpha = 1.0 it divides 0 by 0 instead of returning k + 1.
def expected_tokens(alpha: float, k: int) -> float:
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    return expected_tokens(alpha, k) / (k * c + 1.0)


def best_k(alpha: float, c: float, k_max: int) -> int:
    return max(range(k_max + 1), key=lambda k: speedup(alpha, k, c))
