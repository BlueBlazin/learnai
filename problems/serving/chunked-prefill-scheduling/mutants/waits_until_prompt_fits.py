# BUG: there is no chunking: a prompt is scheduled only when all of it fits in the remaining budget, so a prompt longer than the budget is never prefilled and blocks every request behind it.
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
    for seq_id, remaining in [*prefilling, *waiting]:
        if remaining > budget:
            break
        prefill.append((seq_id, remaining))
        budget -= remaining
    return prefill, decode
