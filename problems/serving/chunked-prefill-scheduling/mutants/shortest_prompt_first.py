# BUG: candidates are served shortest remaining prompt first instead of first come, first served, so a long prompt is overtaken by every short one that arrives after it.
def plan(
    running: list,
    prefilling: list[tuple],
    waiting: list[tuple],
    max_batch_tokens: int,
    long_prefill_threshold: int,
) -> tuple[list[tuple], list]:
    decode = list(running[:max_batch_tokens])
    budget = max_batch_tokens - len(decode)
    prefill = []
    for seq_id, remaining in sorted([*prefilling, *waiting], key=lambda c: c[1]):
        if budget == 0:
            break
        n = min(remaining, long_prefill_threshold, budget)
        prefill.append((seq_id, n))
        budget -= n
    return prefill, decode
