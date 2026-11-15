# BUG: forced tokens are taken without looking at logits, but every token is fed to the model in a call of its own as soon as it is generated, so the number of forward calls is the number of tokens, not the number of free steps.
import torch

from whetstone.kit import TinyLM, TokenConstraint


def jump_forward_decode(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = constraint.start()
    generated = []
    logits, cache = None, None
    pending = list(prompt)
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        # Keeps the cache "up to date" at every step, forced or not.
        logits, cache = lm(pending, cache)
        if len(permitted) == 1:
            token = permitted[0]
        else:
            keep = [False] * lm.vocab_size
            for candidate in permitted:
                keep[candidate] = True
            masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
            token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
        pending = [token]
    return generated
