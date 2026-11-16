# BUG: advances the constraint before checking for end-of-sequence; EOS is not a token of the constraint, so advance raises ValueError.
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
        if not permitted:
            break
        logits, cache = lm(pending, cache)
        keep = [False] * lm.vocab_size
        for token in permitted:
            keep[token] = True
        masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
        token = int(torch.argmax(masked))
        generated.append(token)
        state = constraint.advance(state, token)
        if token == eos_id:
            break
        pending = [token]
    return generated
