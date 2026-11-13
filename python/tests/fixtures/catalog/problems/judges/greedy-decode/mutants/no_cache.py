# BUG: correct tokens, but re-feeds the whole sequence at every step instead of using the cache.
import torch


def greedy_decode(lm, prompt, max_new_tokens):
    tokens = list(prompt)
    for _ in range(max_new_tokens):
        logits, _ = lm(tokens)
        tokens.append(int(torch.argmax(logits[-1])))
    return tokens[len(prompt) :]
