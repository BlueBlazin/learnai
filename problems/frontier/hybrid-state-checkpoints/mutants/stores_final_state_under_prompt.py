# BUG: saves the end-of-prompt checkpoint after generation, when the state already contains the generated tokens: the key says 'the prompt' and the state is of a longer sequence.
import torch

from whetstone.kit import RecurrentLM, RecurrentState

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
        stop = min(len(prompt), (done // interval + 1) * interval)
        logits, state = lm(prompt[done:stop], state)
        done = stop
        if done < len(prompt):
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

    save(prompt, state)
    while len(new_store) > max_checkpoints:
        del new_store[next(iter(new_store))]
    return generated, new_store
