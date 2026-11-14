# BUG: decoded sequences are not counted against the budget: prompt chunks may use all of max_batch_tokens on top of the decodes, so a step can exceed the budget.
def plan(
    running: list,
    prefilling: list[tuple],
    waiting: list[tuple],
    max_batch_tokens: int,
    long_prefill_threshold: int,
) -> tuple[list[tuple], list]:
    decode = list(running[:max_batch_tokens])
    budget = max_batch_tokens
    prefill = []
    for seq_id, remaining in [*prefilling, *waiting]:
        if budget == 0:
            break
        n = min(remaining, long_prefill_threshold, budget)
        prefill.append((seq_id, n))
        budget -= n
    return prefill, decode
