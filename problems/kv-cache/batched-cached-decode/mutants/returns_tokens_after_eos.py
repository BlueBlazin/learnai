# BUG: has no per-row finished flag: every row keeps appending whatever the model produces until all rows have emitted EOS at some point, so rows that finished early carry tokens after their EOS.
import torch

from learnai.kit import TinyLM


def batched_greedy(
    lm: TinyLM, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    logits, cache = lm(prompts)
    rows: list[list[int]] = [[] for _ in prompts]
    while True:
        tokens = torch.argmax(logits[:, -1, :], dim=-1).tolist()
        for b, token in enumerate(tokens):
            rows[b].append(token)
        done = [eos_id in row or len(row) == max_new_tokens for row in rows]
        if all(done):
            return rows
        feed = [[eos_id] if eos_id in rows[b] else [rows[b][-1]] for b in range(len(rows))]
        logits, cache = lm(feed, cache)
