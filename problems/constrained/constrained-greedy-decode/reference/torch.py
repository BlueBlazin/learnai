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
        # What may come next: the tokens that keep the output inside the language,
        # and end-of-sequence if the output is complete as it stands.
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        # Nothing can follow: stop before spending a forward call on it.
        if not permitted:
            break
        logits, cache = lm(pending, cache)
        # Mask: every token that is not permitted gets -inf, so the argmax is the
        # best permitted token (and the lowest id among equal logits).
        keep = [False] * lm.vocab_size
        for token in permitted:
            keep[token] = True
        masked = torch.where(torch.tensor(keep), logits[-1], float("-inf"))
        token = int(torch.argmax(masked))
        generated.append(token)
        if token == eos_id:
            break
        # The constraint must see what was generated, or it keeps answering for the
        # same position.
        state = constraint.advance(state, token)
        pending = [token]
    return generated
