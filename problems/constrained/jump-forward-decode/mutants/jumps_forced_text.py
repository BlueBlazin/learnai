# BUG: jumps whenever the next text is forced, not the next token: if every permitted token spells a prefix of the longest one, it takes the longest without asking the model, choosing a tokenization the model might not have chosen.
import torch

from whetstone.kit import TinyLM, TokenConstraint


def jump_forward_decode(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = constraint.start()
    generated = []
    cache = None
    pending = list(prompt)
    pieces = getattr(getattr(constraint, "vocab", None), "pieces", None)
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        longest = None
        if pieces is not None and eos_id not in permitted:
            longest = max(permitted, key=lambda t: len(pieces[t]))
            if not all(pieces[longest].startswith(pieces[t]) for t in permitted):
                longest = None
        if len(permitted) == 1:
            token = permitted[0]
        elif longest is not None:
            token = longest
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
