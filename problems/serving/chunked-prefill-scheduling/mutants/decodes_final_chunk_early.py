# BUG: a sequence whose last prompt chunk is scheduled in this step is also put in `decode`; it produces its first token from that chunk and can be decoded only from the next step on.
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
        n = min(remaining, long_prefill_threshold, budget)
        prefill.append((seq_id, n))
        budget -= n
        if n == remaining and budget > 0:
            decode.append(seq_id)
            budget -= 1
    return prefill, decode
