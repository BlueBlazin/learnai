import torch

from learnai.kit import TinyLM


def greedy_decode(
    lm: TinyLM, prompt: list[int], max_new_tokens: int, eos_id: int | None
) -> list[int]:
    # Prefill: the whole prompt in one call. The last row of logits scores the first
    # new token, and the cache now holds keys and values for every prompt token.
    logits, cache = lm(prompt)
    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        # Stop before feeding the final token: its logits would never be used.
        if token == eos_id or len(generated) == max_new_tokens:
            return generated
        # Decode: only the new token goes in; the cache supplies the rest of the prefix.
        logits, cache = lm([token], cache)
