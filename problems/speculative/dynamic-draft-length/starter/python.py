import math


def should_stop_drafting(logits: list[float], threshold: float) -> bool:
    """Whether the draft model should stop drafting after the token it just produced.

    logits: the draft model's next-token logits for that token (a list of floats).
    The draft is greedy; its confidence is the softmax probability of the argmax.
    Returns True when the confidence is strictly below `threshold`.
    """
    raise NotImplementedError


def next_k(k: int, n_accepted: int, n_drafted: int) -> int:
    """The draft length for the next round, after a round that drafted n_drafted >= 1
    tokens of which the first n_accepted were accepted.
    """
    raise NotImplementedError


def simulate(
    trace: list[tuple[bool, list[float]]],
    k: int,
    schedule: str,
    threshold: float,
    c_draft: int,
    c_target: int,
) -> dict:
    """Run the draft-and-verify rounds over a trace and total up the cost.

    trace[t] = (match, logits) for output position t: whether the draft's token there
    equals the target's, and the draft's logits for it.
    k: the initial draft length, >= 1. schedule: "constant" or "heuristic".
    threshold: the confidence threshold of should_stop_drafting (0.0 disables it).
    c_draft, c_target: cost of one drafted token and of one target call (ints).
    Returns {"cost": int, "rounds": int, "draft_tokens": int, "ks": list[int]}.
    """
    raise NotImplementedError
