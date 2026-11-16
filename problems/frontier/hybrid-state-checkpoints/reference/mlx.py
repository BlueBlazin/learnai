import mlx.core as mx

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
    new_store = dict(store)  # same order: least recently used first

    def save(sequence, state):
        # A saved (or saved again) checkpoint becomes the most recent entry.
        key = tuple(sequence)
        new_store.pop(key, None)
        new_store[key] = state

    # The deepest checkpoint whose key is a proper prefix of the prompt. A state
    # cannot be cut back, so an entry that goes on differently is useless, and one
    # prompt token must be left to feed: logits come only out of a model call.
    best = None
    for key in store:
        usable = len(key) < len(prompt) and tuple(prompt[: len(key)]) == key
        if usable and (best is None or len(key) > len(best)):
            best = key
    state, done = None, 0
    if best is not None:
        state, done = store[best], len(best)
        save(best, state)  # used: move it to the recent end

    # Prefill in segments that end at the multiples of `interval` and at the end of
    # the prompt: a state can only be saved at the end of a call.
    while done < len(prompt):
        stop = min(len(prompt), (done // interval + 1) * interval)
        logits, state = lm(prompt[done:stop], state)
        done = stop
        save(prompt[:done], state)

    sequence, generated = list(prompt), []
    while True:
        token = int(mx.argmax(logits[-1]).item())
        generated.append(token)
        if len(generated) == max_new_tokens:
            break
        sequence.append(token)
        logits, state = lm([token], state)
        if len(sequence) % interval == 0:
            save(sequence, state)

    # The budget: drop the least recently used entries.
    while len(new_store) > max_checkpoints:
        del new_store[next(iter(new_store))]
    return generated, new_store
