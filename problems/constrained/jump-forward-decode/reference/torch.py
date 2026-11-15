import torch

from whetstone.kit import TinyLM, TokenConstraint


def jump_forward_decode(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = constraint.start()
    generated = []
    cache = None
    pending = list(prompt)  # tokens the model has not seen yet: they wait for a free step
    while len(generated) < max_new_tokens:
        # The permitted set depends on the constraint alone, so it is known before
        # (and without) running the model.
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        if len(permitted) == 1:
            # Forced: the masked argmax could only return this token, whatever the
            # logits are. Take it and let it wait in `pending`.
            token = permitted[0]
        else:
            # Free: the model has to choose. One call catches it up on everything
            # it has not seen (the prompt, or the last free token and the forced
            # run after it), and the last row scores the choice.
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
    # Whatever is still pending is never fed: nothing is chosen after it.
    return generated
