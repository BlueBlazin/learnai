# BUG: prefills the prompt in a call of its own before the loop; when the output begins with forced tokens that call's logits are never used, and the first free step needs a second call.
import torch

from whetstone.kit import TinyLM, TokenConstraint


def jump_forward_decode(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = constraint.start()
    generated = []
    logits, cache = lm(prompt)
    fresh = True  # `logits` already scores the next token
    pending = []
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        if len(permitted) == 1:
            token = permitted[0]
        else:
            if not fresh:
                logits, cache = lm(pending, cache)
                pending = []
            keep = [False] * lm.vocab_size
            for candidate in permitted:
                keep[candidate] = True
            masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
            token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
        pending.append(token)
        fresh = False
    return generated
