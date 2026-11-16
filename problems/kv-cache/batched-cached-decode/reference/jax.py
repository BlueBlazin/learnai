import jax.numpy as jnp

from learnai.kit import TinyLM


def batched_greedy(
    lm: TinyLM, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    # Prefill: all B prompts in one call. logits is (B, P, V); the last position of
    # every row scores that row's first new token.
    logits, cache = lm(prompts)
    rows: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    while True:
        tokens = jnp.argmax(logits[:, -1, :], axis=-1).tolist()
        for b, token in enumerate(tokens):
            if finished[b]:
                continue  # this row's logits come from filler input: ignore them
            rows[b].append(token)
            # A row is finished for good once it emits EOS or uses up its budget.
            finished[b] = token == eos_id or len(rows[b]) == max_new_tokens
        # Stop before feeding anything once no row needs another token.
        if all(finished):
            return rows
        # The batch stays rectangular: a finished row is fed eos_id as filler. (A row
        # can only be finished here if it emitted EOS: rows that run out of budget do
        # so together, and then the loop has already returned.)
        feed = [[eos_id] if finished[b] else [rows[b][-1]] for b in range(len(rows))]
        logits, cache = lm(feed, cache)
