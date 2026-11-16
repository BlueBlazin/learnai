# BUG: feeds the prompt one token per call instead of prefilling it in a single call; the tokens are right but it makes len(prompt) - 1 extra forward calls.
import torch

from learnai.kit import TinyLM


def greedy_decode(
    lm: TinyLM, prompt: list[int], max_new_tokens: int, eos_id: int | None
) -> list[int]:
    cache = None
    for token in prompt:
        logits, cache = lm([token], cache)
    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if token == eos_id or len(generated) == max_new_tokens:
            return generated
        logits, cache = lm([token], cache)
