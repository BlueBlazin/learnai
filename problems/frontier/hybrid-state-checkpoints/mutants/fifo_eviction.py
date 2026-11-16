# BUG: does not move the checkpoint it resumed from to the recent end of the store, so eviction is first-in-first-out and a system prompt's checkpoint that every request uses is evicted as if it were never touched.
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

    while done < len(prompt):
        stop = min(len(prompt), (done // interval + 1) * interval)
        logits, state = lm(prompt[done:stop], state)
        done = stop
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
