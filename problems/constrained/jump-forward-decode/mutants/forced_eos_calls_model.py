# BUG: never jumps in an accepting state: when the output is complete and nothing can follow, EOS is the only permitted token, but the model is still called to 'choose' it.
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
        permitted = list(constraint.allowed(state))
        accepting = constraint.is_accepting(state)
        if accepting:
            permitted.append(eos_id)
        if not permitted:
            break
        if len(permitted) == 1 and not accepting:
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
