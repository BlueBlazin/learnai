# BUG: the budget is given to prompt chunks first and decodes get what is left, so running sequences stall for as long as prompts can fill the budget.
def plan(
    running: list,
    prefilling: list[tuple],
    waiting: list[tuple],
    max_batch_tokens: int,
    long_prefill_threshold: int,
) -> tuple[list[tuple], list]:
    budget = max_batch_tokens
    prefill = []
    for seq_id, remaining in [*prefilling, *waiting]:
        if budget == 0:
            break
        n = min(remaining, long_prefill_threshold, budget)
        prefill.append((seq_id, n))
        budget -= n
    decode = list(running[:budget])
    return prefill, decode
