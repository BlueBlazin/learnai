def expected_tokens(alpha: float, k: int) -> float:
    # The round emits the accepted prefix plus one token from the target. Draft i is
    # reached and accepted with probability alpha**i, so the expectation is the
    # geometric series 1 + alpha + ... + alpha**k.
    if alpha == 1.0:
        # Every draft is accepted; the closed form below would be 0 / 0.
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, c: float) -> float:
    # One round costs k draft calls at c each plus one target call, in units of a
    # target call; plain decoding gets exactly one token per unit.
    return expected_tokens(alpha, k) / (k * c + 1.0)


def best_k(alpha: float, c: float, k_max: int) -> int:
    # k is a small integer, so search it. `max` returns the first maximum it meets,
    # which is the smallest k; k = 0 (no speculation) is a candidate with speedup 1.
    return max(range(k_max + 1), key=lambda k: speedup(alpha, k, c))
