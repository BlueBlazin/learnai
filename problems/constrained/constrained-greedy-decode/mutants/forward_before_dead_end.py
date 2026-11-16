# BUG: runs the model before looking at what the constraint permits, so a step at which nothing is permitted still costs a forward call.
import torch

from learnai.kit import TinyLM, TokenConstraint


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
        logits, cache = lm(pending, cache)
        if not permitted:
            break
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
