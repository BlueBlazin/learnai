"""Tests for early-exit-self-speculation.

Every case checks the generated tokens first, then the cost. Both expectations come
from `simulate`, which runs the round procedure of the statement with no cache at all
(every step recomputes the whole sequence): it returns the tokens, which must equal
the full model's own greedy decode, and what each round drafted and accepted. With R
rounds and T = the sum over rounds of (drafts in the round + 1), the limits are

    layer_calls   <= E * T + (L - E) * R
    layer_tokens  <= L * (P - 1 + T)

`simulate` also verifies what the statement promises about the fixtures: wherever an
argmax is used, at the exit or after the last layer, the two largest logits are at
least 1e-3 apart, so the way calls are split cannot change a token or a round.

The model is the kit's `LayeredLM` with one local change (`GainLM`): the gain of the
final norm is not all ones. With a unit gain the norm only rescales a state, the
argmax of the head is the same with and without it, and forgetting `lm.norm` at the
exit would be invisible.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import LayeredLM

FN = "self_speculative_greedy"
MARGIN = 1e-3
REFINING = (1.0, 1.0, 0.1, 0.1)  # late layers refine: the exit usually agrees

# The worked example of the statement.
PROMPT = [0, 17, 42, 5]
EXAMPLE_TOKENS = [32, 46, 15, 30, 9, 9, 9, 9, 53, 52, 24, 46]
EXAMPLE_ROUNDS = [(4, 3), (4, 1), (4, 1), (3, 3)]  # (drafted, accepted) per round


class GainLM(LayeredLM):
    """A `LayeredLM` whose final RMSNorm has a seeded, non-uniform gain."""

    def __init__(self, fw, seed=0, **sizes):
        super().__init__(fw, seed, **sizes)
        gain = np.random.default_rng([seed, 77]).uniform(0.4, 1.6, self.d_model)
        self._weights = {**self._weights, "norm_out": gain.astype(np.float32)}


_models = {}


def model(ctx, seed, scale, fw, n_layers=4):
    """A fresh twin (own counters) of the model for (seed, scale), speaking `fw`."""
    key = (seed, scale, n_layers)
    if key not in _models:
        _models[key] = GainLM(ctx.oracle_fw, seed, residual_scale=scale, n_layers=n_layers)
    return _models[key].with_framework(fw)


def simulate(ctx, lm, prompt, exit_layer, k, max_new_tokens):
    """The statement's rounds without a cache. Returns (tokens, [(drafted, accepted)])."""
    fw = ctx.oracle_fw

    def rows(seq):
        """Exit logits and full logits at every position of `seq`."""
        h = lm.embed(seq)
        exit_states, cache = lm.layers(h, 0, exit_layer)
        full, _ = lm.layers(exit_states, exit_layer, lm.n_layers, cache)
        return (
            fw.numpy(lm.head(lm.norm(exit_states))),
            fw.numpy(lm.head(lm.norm(full))),
        )

    def clear(logits):
        top = np.sort(logits, axis=-1)[..., -2:]
        return float((top[..., 1] - top[..., 0]).min()) >= MARGIN

    seq, generated, rounds = list(prompt), [], []
    while True:
        n_draft = min(k, max_new_tokens - len(generated) - 1)
        drafts = []
        for _ in range(n_draft):
            early, _ = rows(seq + drafts)
            if not clear(early[-1]):
                raise RuntimeError("authoring error: a near-tie in the exit logits")
            drafts.append(int(early[-1].argmax()))
        _, full = rows(seq + drafts)
        full = full[len(seq) - 1 :]
        greedy = full.argmax(axis=-1).tolist()
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        if not clear(full[: n + 1]):
            raise RuntimeError("authoring error: a near-tie in the full model's logits")
        new = [*drafts[:n], greedy[n]]
        rounds.append((n_draft, n))
        generated += new
        seq += new
        if len(generated) == max_new_tokens:
            return generated, rounds


def full_greedy(ctx, lm, prompt, max_new_tokens):
    """The definition of the output: the full model's greedy decode."""
    seq = list(prompt)
    for _ in range(max_new_tokens):
        logits, _ = lm(seq)
        seq.append(int(ctx.oracle_fw.numpy(logits)[-1].argmax()))
    return seq[len(prompt) :]


def decode(ctx, seed, scale, prompt, exit_layer, k, max_new_tokens, expected=None, n_layers=4):
    """Run the solution; check its tokens, then its cost. Returns (tokens, rounds)."""
    oracle = model(ctx, seed, scale, ctx.oracle_fw, n_layers)
    want, rounds = simulate(ctx, oracle, prompt, exit_layer, k, max_new_tokens)
    if want != full_greedy(ctx, oracle, prompt, max_new_tokens) or (
        expected is not None and want != expected
    ):
        raise RuntimeError("authoring error: the rounds do not reproduce the expected tokens")
    lm = model(ctx, seed, scale, ctx.fw, n_layers)
    got = ctx.call(FN, lm, list(prompt), exit_layer, k, max_new_tokens)
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{FN}: must return a list of Python ints",
        f"got {got!r}",
    )
    check.equal(got, want, label="generated tokens")
    scored = sum(drafted + 1 for drafted, _ in rounds)  # T
    layers = lm.n_layers
    check.cost(
        lm.counters,
        "layer_calls",
        at_most=exit_layer * scored + (layers - exit_layer) * len(rounds),
        unit="layer calls",
    )
    check.cost(
        lm.counters,
        "layer_tokens",
        at_most=layers * (len(prompt) - 1 + scored),
        unit="layer-tokens",
    )
    return got, rounds


@case("the worked example: 12 tokens in 4 rounds, exit after layer 2 of 4, k = 4", public=True)
def worked_example(ctx):
    _, rounds = decode(ctx, 3, REFINING, PROMPT, 2, 4, 12, expected=EXAMPLE_TOKENS)
    if rounds != EXAMPLE_ROUNDS:
        raise RuntimeError("authoring error: the rounds of the statement changed")


@case("an exit that usually agrees: 40 tokens in fewer than half as many rounds", public=True)
def high_agreement(ctx):
    # Rejects exit_not_normalized (right tokens, too many rounds), recomputes_early_layers,
    # verifies_one_by_one, and rebuilds_cache.
    for seed, prompt, k in [(8, [0, 8, 8, 8], 4), (9, [0, 12, 40], 6)]:
        got, rounds = decode(ctx, seed, REFINING, prompt, 2, k, 40)
        if len(rounds) >= len(got) / 2:
            raise RuntimeError("authoring error: this fixture needs too many rounds")


@case("an exit that is usually wrong: every layer's cache is rolled back round after round",
      public=True)  # fmt: skip
def low_agreement(ctx):
    # Rejects no_rollback and late_cache_not_rolled_back (wrong tokens).
    _, rounds = decode(ctx, 5, 1.0, [0, 33, 2, 60, 14, 7], 2, 3, 30)
    accepted = [n for _, n in rounds]
    if not (accepted.count(0) >= 5 and any(0 < n < 3 for n in accepted)):
        raise RuntimeError("authoring error: this fixture no longer mixes rejections")


@case("max_new_tokens falls inside a block: nothing is drafted that cannot be emitted")
def budget_inside_block(ctx):
    # Rejects drafts_past_budget (right tokens, wasted layer calls).
    for limit in (7, 6, 2, 9):
        got, rounds = decode(ctx, 9, REFINING, PROMPT, 2, 4, limit)
        if len(got) != limit or rounds[-1][0] == 4:
            raise RuntimeError("authoring error: the last round is not cut short")


@case("max_new_tokens = 1: all layers once over the prompt, and no draft")
def single_new_token(ctx):
    decode(ctx, 9, REFINING, PROMPT, 2, 4, 1)
    decode(ctx, 3, 1.0, [0, 12, 40, 7, 7], 1, 6, 1)


@case("exit after layer 1 and after layer 3 of 4")
def exit_layers(ctx):
    decode(ctx, 9, REFINING, PROMPT, 1, 4, 24)
    decode(ctx, 9, REFINING, PROMPT, 3, 4, 24)
    decode(ctx, 6, (1.0, 0.1, 0.1, 0.1), [0, 5, 6], 1, 3, 20)


@case("draft lengths k = 1 and k = 8")
def draft_lengths(ctx):
    decode(ctx, 8, REFINING, [0, 8, 8, 8], 2, 1, 20)
    decode(ctx, 8, REFINING, [0, 8, 8, 8], 2, 8, 45)


@case("a six-layer model, exit after layer 4, and a one-token prompt")
def deeper_model(ctx):
    scale = (1.0, 1.0, 1.0, 1.0, 0.1, 0.1)
    decode(ctx, 5, scale, [0], 4, 4, 30, n_layers=6)
    decode(ctx, 5, scale, [0, 9, 9], 2, 4, 20, n_layers=6)


@case("a 40-token prompt and 80 new tokens with k = 8: the whole context window is used")
def long_prompt(ctx):
    # P + max_new_tokens + k is exactly max_context. Each prompt token may pass
    # through each layer once: the limits leave no room for feeding it again.
    decode(ctx, 8, REFINING, [0, *range(20, 59)], 2, 8, 80)
