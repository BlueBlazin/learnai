# BUG: the series stops at alpha**(k-1): it counts k tokens at most and forgets the token the target's pass always adds (the correction or the bonus).
def expected_tokens(alpha: float, k: int) -> float:
    if alpha == 1.0:
        return float(k)
    return (1.0 - alpha**k) / (1.0 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    return expected_tokens(alpha, k) / (k * c + 1.0)


def best_k(alpha: float, c: float, k_max: int) -> int:
    return max(range(k_max + 1), key=lambda k: speedup(alpha, k, c))
