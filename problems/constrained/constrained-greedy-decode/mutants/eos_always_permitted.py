# BUG: permits end-of-sequence at every step instead of only in an accepting state, so the output can stop in the middle of the structure.
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
        permitted.append(eos_id)
        if not permitted:
            break
        logits, cache = lm(pending, cache)
        keep = [False] * lm.vocab_size
        for token in permitted:
            keep[token] = True
        masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
        token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
        pending = [token]
    return generated
