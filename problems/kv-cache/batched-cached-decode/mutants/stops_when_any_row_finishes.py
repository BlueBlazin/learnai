# BUG: ends the loop as soon as one row has finished (any instead of all), so the other rows are cut short.
import torch

from learnai.kit import TinyLM


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
        if any(finished):
            return rows
        feed = [[eos_id] if finished[b] else [rows[b][-1]] for b in range(len(rows))]
        logits, cache = lm(feed, cache)
