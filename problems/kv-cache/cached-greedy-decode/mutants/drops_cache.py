# BUG: feeds each new token without passing the cache, so the model sees a one-token sequence and has forgotten the prefix.
import torch

from learnai.kit import TinyLM


def greedy_decode(
    lm: TinyLM, prompt: list[int], max_new_tokens: int, eos_id: int | None
) -> list[int]:
    logits, _ = lm(prompt)
    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if token == eos_id or len(generated) == max_new_tokens:
            return generated
        logits, _ = lm([token])
