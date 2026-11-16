"""Tests for prefix-cache-reuse.

A case is a sequence of requests against one model. Each request is judged on its own:
first the generated tokens, against greedy decoding done here without any cache; then
the returned store (old entries kept, the new entry covering exactly the prompt and
usable to continue from); then the cost. For a prompt of `P` tokens of which `reused`
come from the store, and `g` generated tokens, the limits are `g` forward calls and
`P - reused + g - 1` tokens processed.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import KVCache, TinyLM

FN = "generate_with_store"

# The worked example of the statement, on TinyLM(seed=3).
SYSTEM = [0, 17, 42, 5, 9]
PROMPT_A = [*SYSTEM, 33, 12, 60]
PROMPT_B = [*SYSTEM, 21, 7, 50, 8]
TOKENS_A = [4, 61, 60, 26]
TOKENS_B = [35, 45, 60, 60]


def common_prefix_len(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def uncached_greedy(ctx, seed, prompt, max_new_tokens):
    """The definition: the whole sequence through the model at every step."""
    lm = TinyLM(ctx.oracle_fw, seed=seed)
    sequence, generated = list(prompt), []
    for _ in range(max_new_tokens):
        row = ctx.oracle_fw.numpy(lm(sequence)[0])[-1]
        top = np.sort(row)[-2:]
        if top[1] - top[0] < 1e-3:
            raise RuntimeError("authoring error: the two largest logits are too close")
        generated.append(int(np.argmax(row)))
        sequence.append(generated[-1])
    return generated


def stored(ctx, seed, *keys):
    """A store built by the test: each key prefilled in one call."""
    lm = TinyLM(ctx.oracle_fw, seed=seed)
    return {tuple(key): lm(list(key))[1] for key in keys}


class Session:
    """One model and the store that the solution hands back, request after request."""

    def __init__(self, ctx, seed, store=None):
        self.ctx = ctx
        self.seed = seed
        self.lm = TinyLM(ctx.fw, seed=seed)
        self.probe_lm = TinyLM(ctx.oracle_fw, seed=seed)
        self.store = dict(store or {})

    def request(self, prompt, max_new_tokens, expected=None, reused=None):
        """Run one request, check it, and adopt the returned store. Returns the tokens."""
        ctx, prompt = self.ctx, [int(t) for t in prompt]
        if expected is None:
            expected = uncached_greedy(ctx, self.seed, prompt, max_new_tokens)
        shared = max((common_prefix_len(key, prompt) for key in self.store), default=0)
        can_reuse = min(shared, len(prompt) - 1)
        if reused is not None and reused != can_reuse:
            raise RuntimeError(f"authoring error: {can_reuse} tokens are reusable, not {reused}")
        setup = (
            f"store keys: {list(self.store)}\nprompt: {prompt}\n"
            f"max_new_tokens: {max_new_tokens}\nreusable prefix: {can_reuse} tokens\n"
        )

        self.lm.counters.reset()
        result = ctx.call(FN, self.lm, dict(self.store), list(prompt), max_new_tokens)
        check.that(
            isinstance(result, tuple) and len(result) == 2,
            f"{FN}: must return a tuple (tokens, store)",
            f"got {type(result).__name__}",
        )
        tokens, new_store = result

        # 1. The tokens.
        check.that(
            isinstance(tokens, list) and all(type(token) is int for token in tokens),
            "tokens: must be a list of Python ints",
            f"got {tokens!r}",
        )
        check.equal(tokens, expected, label="generated tokens")

        # 2. The store.
        key = tuple(prompt)
        check.that(isinstance(new_store, dict), "store: must be a dict")
        check.that(
            set(new_store) == {*self.store, key},
            "store: its keys must be the old keys plus tuple(prompt)",
            f"{setup}returned keys: {list(new_store)}\n",
        )
        for old_key, cache in self.store.items():
            if old_key != key:
                check.that(new_store[old_key] is cache, "store: an old entry was replaced", setup)
        entry = new_store[key]
        check.that(isinstance(entry, KVCache), "store: the new entry must be a KVCache")
        check.that(
            entry.length == len(prompt),
            "store: the new entry must cover exactly the prompt",
            f"{setup}the new entry's cache.length is {entry.length}\n",
        )
        fw = ctx.oracle_fw
        continued = fw.numpy(self.probe_lm([expected[0]], entry)[0])[-1]
        direct = fw.numpy(self.probe_lm([*prompt, expected[0]])[0])[-1]
        check.allclose(
            continued, direct, rtol=1e-3, atol=1e-4, label="logits of a token fed after the new entry"
        )

        # 3. The cost.
        check.cost(self.lm.counters, "forward_calls", at_most=max_new_tokens)
        limit = len(prompt) - can_reuse + max_new_tokens - 1
        check.cost(self.lm.counters, "tokens_processed", at_most=limit, unit="tokens")

        self.store = new_store
        return tokens


@case("the worked example: an empty store, then a prompt sharing 5 tokens", public=True)
def worked_example(ctx):
    session = Session(ctx, 3)
    session.request(PROMPT_A, 4, expected=TOKENS_A, reused=0)  # 8 + 3 = 11 tokens
    # The entry for A continues differently after the 5 shared tokens: truncate it.
    session.request(PROMPT_B, 4, expected=TOKENS_B, reused=5)  # 4 + 3 = 7 tokens


@case("a follow-up that extends a stored prompt reuses the whole entry", public=True)
def extends_entry(ctx):
    session = Session(ctx, 3)
    reply = session.request(PROMPT_A, 4, reused=0)
    session.request([*PROMPT_A, *reply, 30, 31, 2], 5, reused=8)


@case("the same prompt again: all but its last token is reused, and one token is fed", public=True)
def identical_prompt(ctx):
    session = Session(ctx, 3)
    session.request(PROMPT_A, 4, reused=0)
    session.request(PROMPT_A, 4, reused=7)  # 1 + 3 = 4 tokens


@case("a prompt that is a proper prefix of a stored entry")
def prefix_of_entry(ctx):
    # The entry covers more than the prompt; P - 1 tokens of it are reusable.
    key = [0, 40, 41, 12, 12, 9, 55, 3, 28, 6]
    session = Session(ctx, 5, stored(ctx, 5, key))
    session.request(key[:6], 6, reused=5)
    session.request(key[:1], 3, reused=0)


@case("several entries: the one with the longest common prefix is used")
def longest_match(ctx):
    base = [0, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    store = stored(
        ctx,
        7,
        [*base[:2], 60, 61, 62],  # shares 2 tokens, and comes first
        [*base[:9], 50, 51],  # shares 9
        [*base[:5], 44, 45, 46, 47],  # shares 5
        [9, *base[1:]],  # shares none
    )
    session = Session(ctx, 7, store)
    session.request(base, 5, reused=9)
    session.request([*base[:7], 13], 3, reused=7)


@case("no entry shares the first token: a full prefill, and the store keeps its entries")
def nothing_shared(ctx):
    session = Session(ctx, 5, stored(ctx, 5, [3, 10, 11, 12], [4, 10, 11, 12]))
    session.request([0, 10, 11, 12, 13], 4, reused=0)
    if len(session.store) != 3:
        raise RuntimeError("authoring error: three entries expected")


@case("divergence right after the first token, and at the last prompt token")
def divergence_depths(ctx):
    key = [0, 15, 16, 17, 18, 19, 20]
    session = Session(ctx, 9, stored(ctx, 9, key))
    session.request([0, 50, 16, 17, 18], 4, reused=1)
    session.request([*key[:6], 63], 4, reused=6)
    session.request([*key, 1, 2], 2, reused=7)


@case("a single-token prompt, and max_new_tokens = 1")
def smallest(ctx):
    session = Session(ctx, 11, stored(ctx, 11, [0, 9, 8]))
    session.request([0], 3, reused=0)
    session.request([0, 9, 8, 7], 1, reused=3)
    session.request([0, 9, 8, 7], 1, reused=3)


@case("a five-turn conversation on another model: each turn reuses the previous prompt")
def conversation(ctx):
    session = Session(ctx, 4)
    history = [0, 3, 20, 21, 22, 23, 24, 25, 4]  # a system prompt
    previous = []
    for turn in range(5):
        user = [3, *(int(t) for t in ctx.rng.integers(5, 64, size=4 + turn)), 4]
        prompt = [*history, *user]
        # The store holds the previous prompt, not the reply that followed it.
        reply = session.request(prompt, 6, reused=len(previous))
        previous, history = prompt, [*prompt, *reply]
    # Editing an earlier question forks from the middle of the stored prompts.
    session.request([*previous[:40], 61, 62], 4, reused=40)
