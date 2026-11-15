# BUG: next_k grows the draft whenever at least one token was accepted, not only when all drafted tokens were.
import math


def should_stop_drafting(logits: list[float], threshold: float) -> bool:
    top = max(logits)
    confidence = 1.0 / sum(math.exp(x - top) for x in logits)
    return confidence < threshold


def next_k(k: int, n_accepted: int, n_drafted: int) -> int:
    if n_accepted > 0:
        return k + 2
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
    position = 0
    cost = 0
    draft_tokens = 0
    ks = []
    while position < total:
        ks.append(k)
        limit = min(k, total - position - 1)
        n_drafted = 0
        while n_drafted < limit:
            _, logits = trace[position + n_drafted]
            n_drafted += 1
            if should_stop_drafting(logits, threshold):
                break
        n_accepted = 0
        while n_accepted < n_drafted and trace[position + n_accepted][0]:
            n_accepted += 1
        cost += n_drafted * c_draft + c_target
        draft_tokens += n_drafted
        position += n_accepted + 1
        if schedule == "heuristic" and n_drafted > 0:
            k = next_k(k, n_accepted, n_drafted)
    return {"cost": cost, "rounds": len(ks), "draft_tokens": draft_tokens, "ks": ks}
