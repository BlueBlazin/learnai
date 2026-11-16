# BUG: decides 'forced' from the allowed list alone: a state with one allowed token that is also accepting is jumped over, although EOS is permitted there too and the model might have stopped.
import torch

from learnai.kit import TinyLM, TokenConstraint


def jump_forward_decode(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = constraint.start()
    generated = []
    cache = None
    pending = list(prompt)
    while len(generated) < max_new_tokens:
        allowed = list(constraint.allowed(state))
        permitted = list(allowed)
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        if len(allowed) == 1 or len(permitted) == 1:
            token = permitted[0]
        else:
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
    return generated
