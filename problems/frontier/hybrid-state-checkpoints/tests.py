"""Tests for hybrid-state-checkpoints.

A case is a sequence of requests against one model and one checkpoint store. Each
request is judged on its own, in this order:

1. the generated tokens, against greedy decoding done here on the whole sequence
   without any saved state;
2. the returned store: its size against `max_checkpoints`, then its keys in order
   (least recently used first), then every state, bit for bit, against the state of a
   fresh run over its key;
3. the cost. With `r` tokens covered by the checkpoint that is resumed from, the
   limits are `P - r + g - 1` tokens processed and one forward call per prefill
   segment plus `g - 1`.

`plan` below is the specification of the store, written independently of the
reference solution.

The solution gets the model as `LM`, a `RecurrentLM` behind the kit's `Guarded`: it may
call it, use `initial_state`, and read the sizes. The counters, the weights, the seed,
and `with_framework` are refused, so the counts cannot be written and no token can be
read by a model that does not count it.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import Guarded, RecurrentLM, RecurrentState, check_guard, guard

FN = "generate_with_checkpoints"

# The worked example of the statement, on RecurrentLM(seed=3).
PROMPT_A = [0, 17, 42, 5, 9, 33, 12, 60, 21, 7]
PROMPT_B = [0, 17, 42, 5, 9, 33, 50, 8, 3]
TOKENS_A = [26, 6, 7, 26]
TOKENS_B = [11, 32, 38]


class LM(Guarded, RecurrentLM):
    """The model the solution gets: the call, `initial_state`, and the sizes."""

    label = "lm"
    refused = Guarded.refused | {"seed"}

    def __repr__(self):  # RecurrentLM's shows the seed, which would rebuild the model
        return f"RecurrentLM(vocab_size={self.vocab_size}, max_context={self.max_context})"


def uncached_greedy(ctx, seed, prompt, max_new_tokens):
    """The definition: the whole sequence through the model at every step."""
    lm = RecurrentLM(ctx.oracle_fw, seed=seed)
    sequence, generated = list(prompt), []
    for _ in range(max_new_tokens):
        row = ctx.oracle_fw.numpy(lm(sequence)[0])[-1]
        generated.append(int(np.argmax(row)))
        sequence.append(generated[-1])
    return generated


def plan(keys, prompt, generated, interval, max_checkpoints):
    """What the statement asks for.

    Returns (resumed length, keys of the new store in order, prefill calls, the keys
    that are resumed from or saved in this request).
    """
    usable = [k for k in keys if len(k) < len(prompt) and tuple(prompt[: len(k)]) == k]
    resumed = max(usable, key=len, default=())
    fed = [*prompt, *generated[:-1]]  # the last generated token is never fed
    lengths = {c for c in range(len(resumed) + 1, len(fed) + 1) if c % interval == 0}
    lengths.add(len(prompt))
    order = list(keys)
    touched = ([resumed] if resumed else []) + [tuple(fed[:c]) for c in sorted(lengths)]
    for key in touched:
        if key in order:
            order.remove(key)
        order.append(key)
    prefill_calls = sum(1 for c in lengths if c <= len(prompt))
    return len(resumed), order[-max_checkpoints:], prefill_calls, set(touched)


def stored(ctx, seed, *keys):
    """A store built by the test: each key read by a fresh model in one call."""
    lm = RecurrentLM(ctx.oracle_fw, seed=seed)
    return {tuple(key): lm(list(key))[1] for key in keys}


def same_state(a, b):
    return a.length == b.length and all(
        np.array_equal(x, y) for x, y in zip(a.layers, b.layers, strict=True)
    )


class Session:
    """One model and the store that the solution hands back, request after request."""

    def __init__(self, ctx, seed, interval, max_checkpoints, store=None):
        self.ctx = ctx
        self.seed = seed
        self.interval = interval
        self.max_checkpoints = max_checkpoints
        self.lm = LM(ctx.fw, seed=seed)
        self.probe_lm = RecurrentLM(ctx.oracle_fw, seed=seed)
        self.store = dict(store or {})

    def request(self, prompt, max_new_tokens, expected=None, resumed=None):
        """Run one request, check it, and adopt the returned store. Returns the tokens."""
        ctx, prompt = self.ctx, [int(t) for t in prompt]
        if expected is None:
            expected = uncached_greedy(ctx, self.seed, prompt, max_new_tokens)
        r, want_keys, prefill_calls, touched = plan(
            list(self.store), prompt, expected, self.interval, self.max_checkpoints
        )
        if resumed is not None and resumed != r:
            raise RuntimeError(f"authoring error: the usable checkpoint has {r} tokens")
        setup = (
            f"store keys, least recently used first (lengths {[len(k) for k in self.store]}):\n"
            f"  {list(self.store)}\nprompt: {prompt}\nmax_new_tokens: {max_new_tokens}, "
            f"interval: {self.interval}, max_checkpoints: {self.max_checkpoints}\n"
            f"deepest stored checkpoint that is a proper prefix of the prompt: {r} tokens\n"
        )

        self.lm.counters.reset()
        guard(ctx)
        result = ctx.call(
            FN,
            self.lm,
            dict(self.store),
            list(prompt),
            max_new_tokens,
            self.interval,
            self.max_checkpoints,
        )
        check_guard()
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

        # 2. The store: size, keys and their order, then the states.
        check.that(isinstance(new_store, dict), "store: must be a dict")
        check.at_most(
            "checkpoints_stored", len(new_store), self.max_checkpoints, unit="checkpoints"
        )
        got_keys = list(new_store)
        detail = (
            f"{setup}expected keys (lengths {[len(k) for k in want_keys]}):\n  {want_keys}\n"
            f"returned keys (lengths {[len(k) for k in got_keys]}):\n  {got_keys}\n"
        )
        check.that(
            all(isinstance(key, tuple) for key in got_keys),
            "store: every key must be a tuple of token ids",
            detail,
        )
        check.that(
            set(got_keys) == set(want_keys),
            "store: the checkpoints it holds are not the ones the statement asks for "
            f"(it has checkpoints of {sorted(len(k) for k in got_keys)} tokens)",
            detail,
        )
        check.that(
            got_keys == want_keys,
            "store: the right checkpoints, but not in least-recently-used-first order",
            detail,
        )
        for key, state in self.store.items():
            if key in new_store and key not in touched:
                check.that(
                    new_store[key] is state,
                    "store: an entry that was neither resumed from nor saved again was replaced",
                    f"{setup}key: {key}\n",
                )
        for key, state in new_store.items():
            check.that(
                isinstance(state, RecurrentState),
                "store: every value must be a RecurrentState that the model returned",
                f"the value under a key of {len(key)} tokens is a {type(state).__name__}",
            )
            check.that(
                same_state(state, self.probe_lm(list(key))[1]),
                f"store: the state saved under a key of {len(key)} tokens is not the state "
                "of the model after exactly those tokens",
                f"{setup}key: {key}\nstate.length: {state.length}\n",
            )

        # 3. The cost.
        check.cost(
            self.lm.counters, "tokens_processed", at_most=len(prompt) - r + max_new_tokens - 1,
            unit="tokens",
        )  # fmt: skip
        check.cost(self.lm.counters, "forward_calls", at_most=prefill_calls + max_new_tokens - 1)

        self.store = new_store
        return tokens


@case("the worked example: an empty store, then a prompt that shares 6 tokens", public=True)
def worked_example(ctx):
    session = Session(ctx, 3, interval=4, max_checkpoints=4)
    session.request(PROMPT_A, 4, expected=TOKENS_A, resumed=0)  # 10 + 3 = 13 tokens
    # Six tokens are shared, but only the checkpoint after four of them is usable.
    session.request(PROMPT_B, 3, expected=TOKENS_B, resumed=4)  # 5 + 2 = 7 tokens
    if [len(k) for k in session.store] != [12, 4, 8, 9]:
        raise RuntimeError("authoring error: the statement's final store")


@case("a follow-up turn resumes from a checkpoint taken while generating", public=True)
def follow_up_turn(ctx):
    session = Session(ctx, 3, interval=4, max_checkpoints=8)
    reply = session.request(PROMPT_A, 7, resumed=0)  # checkpoints at 4, 8, 10, 12, 16
    session.request([*PROMPT_A, *reply, 30, 31, 2], 3, resumed=16)
    # A follow-up that arrives before a multiple of the interval: the end of the prompt.
    session.request([*PROMPT_A, 44, 45], 2, resumed=10)


@case("the same prompt again: the checkpoint of the whole prompt cannot be used", public=True)
def identical_prompt(ctx):
    prompt = [0, 49, 47, 54, 12, 7, 55, 3]
    session = Session(ctx, 3, interval=4, max_checkpoints=8)
    first = session.request(prompt, 4, resumed=0)
    # Keys of 4 and 8 tokens exist. The one of 8 leaves no token to feed.
    again = session.request(prompt, 4, resumed=4)  # 4 + 3 = 7 tokens
    if first != again:
        raise RuntimeError("authoring error: the two runs must agree")


@case("a budget of 3: the least recently used checkpoints go, the reused one stays", public=True)
def budget(ctx):
    system = [0, 20, 21, 22, 23, 24]
    session = Session(ctx, 5, interval=6, max_checkpoints=3)
    for turn in range(4):
        user = [int(t) for t in ctx.rng.integers(5, 64, size=3 + turn)]
        # Every request resumes from the system prompt, which keeps it in the store.
        session.request([*system, *user], 2, resumed=6 if turn else 0)
    if tuple(system) not in session.store:
        raise RuntimeError("authoring error: the system prompt's checkpoint must survive")


@case("an entry that shares more tokens but goes on differently is not a checkpoint")
def longer_entry_diverges(ctx):
    # Catches resuming from the entry with the longest common prefix (the KV habit),
    # and rounding the shared length up to the next stored state.
    base = [0, 40, 41, 12, 12, 9, 55, 3, 28, 6, 17]
    store = stored(ctx, 7, base[:3], [*base[:7], 60, 61], [*base[:9], 13], [7, *base[1:5]])
    session = Session(ctx, 7, interval=5, max_checkpoints=12, store=store)
    session.request(base, 4, resumed=3)
    # Now a checkpoint of 10 tokens exists; a prompt that leaves it at token 9 does not use it.
    session.request([*base[:8], 33, 34], 3, resumed=5)


@case("no stored state is a prefix of the prompt: a full prefill")
def nothing_usable(ctx):
    store = stored(ctx, 5, [3, 10, 11, 12], [0, 10, 11, 12, 13, 14, 15], [0, 11])
    session = Session(ctx, 5, interval=3, max_checkpoints=6, store=store)
    session.request([0, 10, 11, 12, 13, 63], 4, resumed=0)
    # The prompt again: its own checkpoint (6 tokens) is not a proper prefix; 3 is.
    session.request([0, 10, 11, 12, 13, 63], 5, resumed=3)


@case("checkpoints sit at absolute multiples of the interval, wherever the run resumed")
def absolute_positions(ctx):
    # The stored state has 5 tokens; with interval 4 the next checkpoints are at 8 and 12.
    key = [0, 15, 16, 17, 18]
    session = Session(ctx, 9, interval=4, max_checkpoints=10, store=stored(ctx, 9, key))
    session.request([*key, 19, 20, 21, 22, 23, 24], 5, resumed=5)
    session.request([*key, 19, 20, 21, 50, 51, 52, 53, 54], 2, resumed=8)


@case("interval 1, and an interval longer than the whole sequence")
def extreme_intervals(ctx):
    prompt = [0, 33, 12, 60, 8, 21]
    session = Session(ctx, 11, interval=1, max_checkpoints=5)
    session.request(prompt, 4, resumed=0)
    session.request([*prompt[:4], 2, 2, 2], 3, resumed=0)  # the short checkpoints were evicted
    session = Session(ctx, 11, interval=50, max_checkpoints=5)
    reply = session.request(prompt, 4, resumed=0)
    session.request([*prompt, *reply, 9], 2, resumed=6)
    session.request(prompt[:5], 2, resumed=0)


@case("a single-token prompt, max_new_tokens = 1, and a budget of one checkpoint")
def smallest(ctx):
    session = Session(ctx, 13, interval=2, max_checkpoints=1)
    session.request([0], 1, resumed=0)
    session.request([0, 9, 8, 7], 1, resumed=1)
    session.request([0, 9, 8, 7, 6], 3, resumed=4)
    session.request([0, 9, 8, 7, 6], 3, resumed=0)


@case("a five-turn conversation on another model, then an edit of an earlier turn")
def conversation(ctx):
    session = Session(ctx, 4, interval=8, max_checkpoints=6)
    history = [0, 3, 20, 21, 22, 23, 24, 25, 4]  # a system prompt
    prompts = []
    for turn in range(5):
        user = [3, *(int(t) for t in ctx.rng.integers(5, 64, size=4 + turn)), 4]
        prompt = [*history, *user]
        reply = session.request(prompt, 6)
        prompts.append(prompt)
        history = [*prompt, *reply]
    # Editing the third question forks from the middle of what is stored.
    session.request([*prompts[2][:-3], 61, 62], 4)
    # Regenerating the last answer saves checkpoints that already exist.
    session.request(prompts[-1], 6)
