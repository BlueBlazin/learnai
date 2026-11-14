# BUG: a chunk is capped only by the remaining budget, not by long_prefill_threshold, so a long prompt at the head takes everything the decodes leave and short prompts behind it wait until it is done.
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
        if budget == 0:
            break
        n = min(remaining, budget)
        prefill.append((seq_id, n))
        budget -= n
    return prefill, decode
