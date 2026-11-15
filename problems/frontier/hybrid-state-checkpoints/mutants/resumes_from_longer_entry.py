# BUG: treats the store like a KV prefix cache: it takes the entry with the longest common prefix and continues from its state, although that state also contains the entry's tokens after the shared part and cannot be cut back.
import torch

from whetstone.kit import RecurrentLM, RecurrentState

Store = dict[tuple[int, ...], RecurrentState]


def common_prefix_len(a, b) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


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

    best, shared = None, 0
    for key in store:
        n = min(common_prefix_len(key, prompt), len(prompt) - 1)
        if n > shared:
            best, shared = key, n
    state, done = None, 0
    if best is not None:
        state, done = store[best], shared
        save(best, state)

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
