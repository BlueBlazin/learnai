# BUG: keeps feeding a finished row the argmax of its (ignored) logits instead of eos_id, so the model goes on generating a continuation past EOS for that row.
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
        if all(finished):
            return rows
        logits, cache = lm([[token] for token in tokens], cache)
