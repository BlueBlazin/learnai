# BUG: ignores the cache and runs the whole sequence (prompt plus everything generated) through the model at every step; the tokens are right but the cost is quadratic.
import torch

from learnai.kit import TinyLM


def greedy_decode(
    lm: TinyLM, prompt: list[int], max_new_tokens: int, eos_id: int | None
) -> list[int]:
    generated = []
    while True:
        logits, _ = lm(prompt + generated)
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if token == eos_id or len(generated) == max_new_tokens:
            return generated
