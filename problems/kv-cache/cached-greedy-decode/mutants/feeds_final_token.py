# BUG: feeds every generated token to the model before checking whether to stop, so the final token costs one forward call whose logits are never used.
import torch

from learnai.kit import TinyLM


def greedy_decode(
    lm: TinyLM, prompt: list[int], max_new_tokens: int, eos_id: int | None
) -> list[int]:
    logits, cache = lm(prompt)
    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        logits, cache = lm([token], cache)
        if token == eos_id or len(generated) == max_new_tokens:
            return generated
