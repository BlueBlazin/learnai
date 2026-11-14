# BUG: applies the mask by multiplying the logits with 0 or 1, which turns forbidden logits into 0 instead of -inf; a forbidden token wins whenever every permitted logit is negative.
import torch

from whetstone.kit import TinyLM, TokenConstraint


def constrained_greedy(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = constraint.start()
    generated = []
    cache = None
    pending = prompt  # the tokens the model has not seen yet
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        logits, cache = lm(pending, cache)
        keep = [False] * lm.vocab_size
        for token in permitted:
            keep[token] = True
        masked = logits[-1] * torch.tensor(keep)
        token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
        pending = [token]
    return generated
