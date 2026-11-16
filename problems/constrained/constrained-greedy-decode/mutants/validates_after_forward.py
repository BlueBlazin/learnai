# BUG: picks the unconstrained argmax and feeds it to the model before checking it against the constraint; a rejected candidate is masked and the step retried, so every rejection (and the final token) costs an extra forward call.
import torch

from learnai.kit import TinyLM, TokenConstraint


def constrained_greedy(
    lm: TinyLM, prompt: list[int], constraint: TokenConstraint, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = constraint.start()
    generated = []
    logits, cache = lm(prompt)
    while len(generated) < max_new_tokens:
        permitted = list(constraint.allowed(state))
        if constraint.is_accepting(state):
            permitted.append(eos_id)
        if not permitted:
            break
        row = logits[-1].clone()
        while True:
            token = int(torch.argmax(row))
            next_logits, next_cache = lm([token], cache)
            if token in permitted:
                break
            row[token] = float("-inf")
        logits, cache = next_logits, next_cache
        generated.append(token)
        if token == eos_id:
            break
        state = constraint.advance(state, token)
    return generated
