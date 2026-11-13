# BUG: after prefilling the whole prompt it starts the loop by feeding the last prompt token again, so that token is in the context twice and every prediction is one position late.
import torch

from whetstone.kit import TinyLM


def greedy_decode(
    lm: TinyLM, prompt: list[int], max_new_tokens: int, eos_id: int | None
) -> list[int]:
    _, cache = lm(prompt)
    token = prompt[-1]
    generated = []
    while True:
        logits, cache = lm([token], cache)
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if token == eos_id or len(generated) == max_new_tokens:
            return generated
