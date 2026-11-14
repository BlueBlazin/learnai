# BUG: avoids the division by zero by clamping alpha to 1 - 1e-9; near 1 the closed form then subtracts two nearly equal numbers and the result is off by far more than the tolerance.
def expected_tokens(alpha: float, k: int) -> float:
    alpha = min(alpha, 1.0 - 1e-9)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    return expected_tokens(alpha, k) / (k * c + 1.0)


def best_k(alpha: float, c: float, k_max: int) -> int:
    return max(range(k_max + 1), key=lambda k: speedup(alpha, k, c))
