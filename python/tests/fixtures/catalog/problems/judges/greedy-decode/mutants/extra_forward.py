# BUG: runs the model on the final token although its logits are never used.
import torch


def greedy_decode(lm, prompt, max_new_tokens):
    logits, cache = lm(prompt)
    out = []
    for _ in range(max_new_tokens):
        token = int(torch.argmax(logits[-1]))
        out.append(token)
        logits, cache = lm([token], cache)
    return out
