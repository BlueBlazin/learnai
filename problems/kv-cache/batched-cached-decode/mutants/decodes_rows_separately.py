# BUG: runs an independent cached greedy loop per row instead of one batched call per step; the tokens are right but the number of forward calls is the sum of the row lengths, not the longest row.
import torch

from learnai.kit import TinyLM


def batched_greedy(
    lm: TinyLM, prompts: list[list[int]], max_new_tokens: int, eos_id: int | None
) -> list[list[int]]:
    rows = []
    for prompt in prompts:
        logits, cache = lm(prompt)
        row: list[int] = []
        while True:
            token = int(torch.argmax(logits[-1]))
            row.append(token)
            if token == eos_id or len(row) == max_new_tokens:
                break
            logits, cache = lm([token], cache)
        rows.append(row)
    return rows
