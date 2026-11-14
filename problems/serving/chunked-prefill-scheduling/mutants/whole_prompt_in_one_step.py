# BUG: a prompt is never split: every candidate that is reached gets all of its remaining tokens in one chunk, whatever the budget and the threshold allow.
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
        if budget <= 0:
            break
        prefill.append((seq_id, remaining))
        budget -= remaining
    return prefill, decode
