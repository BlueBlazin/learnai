# BUG: stops at eos_id but does not include it in the returned tokens.
import torch

from learnai.kit import TinyLM


def greedy_decode(
    lm: TinyLM, prompt: list[int], max_new_tokens: int, eos_id: int | None
) -> list[int]:
    logits, cache = lm(prompt)
    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        if token == eos_id:
            return generated
        generated.append(token)
        if len(generated) == max_new_tokens:
            return generated
        logits, cache = lm([token], cache)
