def expected_tokens(alpha: float, k: int) -> float:
    """Expected number of tokens one speculative round produces.

    alpha: probability in [0, 1] that a draft token is accepted (may be exactly 1.0).
    k: number of draft tokens per round, >= 0.
    The round keeps the accepted prefix and always adds one token from the target.
    Returns a float: 1 + alpha + ... + alpha**k.
    """
    raise NotImplementedError


def speedup(alpha: float, k: int, c: float) -> float:
    """Expected wall-clock speedup over decoding with the target model alone.

    c: cost of one draft call divided by the cost of one target call, >= 0.
    A round costs k draft calls and one target call.
    Returns a float.
    """
    raise NotImplementedError


def best_k(alpha: float, c: float, k_max: int) -> int:
    """The k in 0..k_max with the largest speedup(alpha, k, c); the smallest on a tie.

    k = 0 means no speculation (speedup exactly 1).
    Returns an int.
    """
    raise NotImplementedError
