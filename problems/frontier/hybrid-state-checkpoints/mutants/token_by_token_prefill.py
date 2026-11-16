# BUG: feeds the prompt one token per call so that a state is at hand at every position: the right tokens and checkpoints, but one forward call per prompt token instead of one per segment.
import torch

from learnai.kit import RecurrentLM, RecurrentState

Store = dict[tuple[int, ...], RecurrentState]


def generate_with_checkpoints(
    lm: RecurrentLM,
    store: Store,
    prompt: list[int],
    max_new_tokens: int,
    interval: int,
    max_checkpoints: int,
) -> tuple[list[int], Store]:
    new_store = dict(store)

    def save(sequence, state):
        key = tuple(sequence)
        new_store.pop(key, None)
        new_store[key] = state

    best = None
    for key in store:
        usable = len(key) < len(prompt) and tuple(prompt[: len(key)]) == key
        if usable and (best is None or len(key) > len(best)):
            best = key
    state, done = None, 0
    if best is not None:
        state, done = store[best], len(best)
        save(best, state)

    while done < len(prompt):
        logits, state = lm(prompt[done : done + 1], state)
        done += 1
        if done % interval == 0 or done == len(prompt):
            save(prompt[:done], state)

    sequence, generated = list(prompt), []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if len(generated) == max_new_tokens:
            break
        sequence.append(token)
        logits, state = lm([token], state)
        if len(sequence) % interval == 0:
            save(sequence, state)

    while len(new_store) > max_checkpoints:
        del new_store[next(iter(new_store))]
    return generated, new_store
