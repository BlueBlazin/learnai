import math


def should_stop_drafting(logits: list[float], threshold: float) -> bool:
    # The draft is greedy, so its token is the argmax and its confidence is the largest
    # softmax probability: exp(top) / sum(exp(x)) = 1 / sum(exp(x - top)). Shifting by
    # the maximum keeps every exponent at or below 0, so nothing overflows.
    top = max(logits)
    confidence = 1.0 / sum(math.exp(x - top) for x in logits)
    return confidence < threshold


def next_k(k: int, n_accepted: int, n_drafted: int) -> int:
    # Every drafted token was accepted: the draft is doing well, so look further ahead.
    if n_accepted == n_drafted:
        return k + 2
    # A token was rejected: back off by one, but never stop drafting altogether.
    return max(1, k - 1)


def simulate(
    trace: list[tuple[bool, list[float]]],
    k: int,
    schedule: str,
    threshold: float,
    c_draft: int,
    c_target: int,
) -> dict:
    total = len(trace)
    position = 0  # tokens generated so far
    cost = 0
    draft_tokens = 0
    ks = []
    while position < total:
        ks.append(k)
        # The target always adds one token, so never draft more than remaining - 1.
        limit = min(k, total - position - 1)
        n_drafted = 0
        while n_drafted < limit:
            _, logits = trace[position + n_drafted]
            n_drafted += 1  # the token is drafted (and paid for) before it is judged
            if should_stop_drafting(logits, threshold):
                break
        # Verification keeps the drafted tokens up to the first mismatch.
        n_accepted = 0
        while n_accepted < n_drafted and trace[position + n_accepted][0]:
            n_accepted += 1
        cost += n_drafted * c_draft + c_target
        draft_tokens += n_drafted
        position += n_accepted + 1  # the accepted prefix plus the target's own token
        if schedule == "heuristic" and n_drafted > 0:
            k = next_k(k, n_accepted, n_drafted)
    return {"cost": cost, "rounds": len(ks), "draft_tokens": draft_tokens, "ks": ks}
