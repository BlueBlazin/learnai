# BUG: checks whether all rows have finished only after the next forward call, so one batched call is spent after every row is done.
import torch

from whetstone.kit import TinyLM


def batched_greedy(
    lm: TinyLM, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    logits, cache = lm(prompts)
    rows: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    while True:
        tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for b, token in enumerate(tokens):
            if finished[b]:
                continue
            rows[b].append(token)
            finished[b] = token == eos_id or len(rows[b]) == max_new_tokens
        feed = [[eos_id] if finished[b] and eos_id is not None else [tokens[b]] for b in range(len(rows))]
        logits, cache = lm(feed, cache)
        if all(finished):
            return rows
