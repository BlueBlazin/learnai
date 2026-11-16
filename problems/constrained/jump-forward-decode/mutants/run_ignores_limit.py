# BUG: jumps over a whole run of forced tokens in an inner loop that does not check max_new_tokens, so a run that crosses the limit makes the output too long.
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
        # Jump: take forced tokens for as long as there is exactly one permitted.
        while True:
            permitted = list(constraint.allowed(state))
            if constraint.is_accepting(state):
                permitted.append(eos_id)
            if len(permitted) != 1:
                break
            generated.append(permitted[0])
            if permitted[0] == eos_id:
                return generated
            state = constraint.advance(state, permitted[0])
            pending.append(permitted[0])
        if not permitted or len(generated) >= max_new_tokens:
            break
        logits, cache = lm(pending, cache)
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
