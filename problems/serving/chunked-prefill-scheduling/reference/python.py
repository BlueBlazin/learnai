def plan(
    running: list,
    prefilling: list[tuple],
    waiting: list[tuple],
    max_batch_tokens: int,
    long_prefill_threshold: int,
) -> tuple[list[tuple], list]:
    # Decodes first: one token each, so running sequences never wait for a prompt.
    decode = list(running[:max_batch_tokens])
    budget = max_batch_tokens - len(decode)

    # What is left goes to prompts, oldest first: sequences already being prefilled,
    # then waiting requests. A chunk is capped by the budget that remains and by the
    # threshold, so one long prompt cannot take the whole step from the ones behind it.
    prefill = []
    for seq_id, remaining in [*prefilling, *waiting]:
        if budget == 0:
            break
        n = min(remaining, long_prefill_threshold, budget)
        prefill.append((seq_id, n))
        budget -= n
    return prefill, decode
