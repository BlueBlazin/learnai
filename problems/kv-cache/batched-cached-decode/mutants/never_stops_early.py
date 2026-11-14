# BUG: always runs max_new_tokens steps, also when every row has already emitted EOS; the tokens are right but the model is called after all rows have finished.
import torch

from whetstone.kit import TinyLM


def batched_greedy(
    lm: TinyLM, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    logits, cache = lm(prompts)
    rows: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    for step in range(max_new_tokens):
        tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for b, token in enumerate(tokens):
            if finished[b]:
                continue
            rows[b].append(token)
            finished[b] = token == eos_id or len(rows[b]) == max_new_tokens
        if step == max_new_tokens - 1:
            break
        feed = [[eos_id] if finished[b] else [rows[b][-1]] for b in range(len(rows))]
        logits, cache = lm(feed, cache)
    return rows
