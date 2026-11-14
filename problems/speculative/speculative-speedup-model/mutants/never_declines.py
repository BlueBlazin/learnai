# BUG: best_k searches k from 1, so it recommends speculating even when every draft length is slower than plain decoding (the answer should be 0).
def expected_tokens(alpha: float, k: int) -> float:
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    return expected_tokens(alpha, k) / (k * c + 1.0)


def best_k(alpha: float, c: float, k_max: int) -> int:
    return max(range(1, max(k_max, 1) + 1), key=lambda k: speedup(alpha, k, c))
