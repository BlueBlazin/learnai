"""Tests for speculative-decode-loop.

Every case checks the generated tokens first, then the cost. Both expectations come
from `simulate`, which runs the round procedure of the statement with no cache at all
(every model call gets the whole sequence): it returns the tokens, which must equal
the target's own greedy decode, and the number of rounds R. The limits are then

    target forward_calls     <= R
    target tokens_processed  <= P + R * (k + 1)
    draft tokens_processed   <= P + R * (k + 1)

`simulate` also verifies what the statement promises about the fixtures: wherever a
model's argmax is used, its two largest logits are at least 1e-3 apart, so the split of
the calls (which changes logits by about 1e-5) cannot change a token or the number of
rounds.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import draft_target_pair

FN = "speculative_greedy"
MARGIN = 1e-3

# The worked example of the statement.
PROMPT = [0, 17, 42, 5]
CONTINUATION = [42, 62, 33, 33, 40, 32, 45, 37, 40, 33, 40, 55]

_pairs = {}


def models(ctx, seed, agreement, fw):
    """A fresh (draft, target) pair for `fw`; the weights are built once per pair."""
    key = (seed, agreement)
    if key not in _pairs:
        _pairs[key] = draft_target_pair(ctx.oracle_fw, seed=seed, agreement=agreement)
    draft, target = _pairs[key]
    return draft.with_framework(fw), target.with_framework(fw)


def simulate(ctx, seed, agreement, prompt, k, max_new_tokens, eos_id):
    """The statement's rounds without any cache. Returns (tokens, accepted per round)."""
    fw = ctx.oracle_fw
    draft, target = models(ctx, seed, agreement, fw)

    def rows(lm, tokens, first):
        logits, _ = lm(tokens)
        logits = fw.numpy(logits)[first:]
        top = np.sort(logits, axis=-1)[:, -2:]
        return logits.argmax(axis=-1).tolist(), top[:, 1] - top[:, 0]

    seq, generated, accepted = list(prompt), [], []
    while True:
        n_draft = min(k, max_new_tokens - len(generated) - 1)
        drafts = []
        for _ in range(n_draft):
            (token,), (gap,) = rows(draft, seq + drafts, -1)
            if gap < MARGIN:
                raise RuntimeError("authoring error: a near-tie in the draft's logits")
            drafts.append(token)
        greedy, gaps = rows(target, seq + drafts, len(seq) - 1)
        n = 0
        while n < n_draft and drafts[n] == greedy[n]:
            n += 1
        if gaps[: n + 1].min() < MARGIN:
            raise RuntimeError("authoring error: a near-tie in the target's logits")
        new = [*drafts[:n], greedy[n]]
        if eos_id in new:
            new = new[: new.index(eos_id) + 1]
        accepted.append(n)
        generated += new
        seq += new
        if new[-1] == eos_id or len(generated) == max_new_tokens:
            return generated, accepted


def target_greedy(ctx, seed, agreement, prompt, max_new_tokens, eos_id):
    """The definition of the output: the target's greedy decode, without the draft."""
    fw = ctx.oracle_fw
    _, target = models(ctx, seed, agreement, fw)
    seq, generated = list(prompt), []
    while len(generated) < max_new_tokens:
        logits, _ = target(seq)
        token = int(fw.numpy(logits)[-1].argmax())
        generated.append(token)
        seq.append(token)
        if token == eos_id:
            break
    return generated


def decode(ctx, seed, agreement, prompt, k, max_new_tokens, eos_id, expected=None):
    """Run the solution; check its tokens, then its cost. Returns (tokens, accepted)."""
    want, accepted = simulate(ctx, seed, agreement, prompt, k, max_new_tokens, eos_id)
    if want != target_greedy(ctx, seed, agreement, prompt, max_new_tokens, eos_id) or (
        expected is not None and want != expected
    ):
        raise RuntimeError("authoring error: the rounds do not reproduce the expected tokens")
    draft, target = models(ctx, seed, agreement, ctx.fw)
    got = ctx.call(FN, target, draft, list(prompt), k, max_new_tokens, eos_id)
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{FN}: must return a list of Python ints",
        f"got {got!r}",
    )
    check.equal(got, want, label="generated tokens")
    rounds = len(accepted)
    limit = len(prompt) + rounds * (k + 1)
    # check.at_most rather than check.cost: the two models' counters share their names.
    counters = target.counters
    check.at_most("target_forward_calls", counters.forward_calls, rounds, unit="calls")
    check.at_most("target_tokens_processed", counters.tokens_processed, limit, unit="tokens")
    check.at_most("draft_tokens_processed", draft.counters.tokens_processed, limit, unit="tokens")
    return got, accepted


@case("the worked example: 12 tokens in 4 target calls, with k = 4", public=True)
def worked_example(ctx):
    _, accepted = decode(ctx, 3, 0.9, PROMPT, 4, 12, None, expected=CONTINUATION)
    if accepted != [0, 4, 4, 0]:
        raise RuntimeError("authoring error: the rounds of the statement changed")


@case("EOS among the accepted drafts: with eos_id = 40 the example ends after 5 tokens",
      public=True)
def eos_inside_block(ctx):
    # Rejects eos_inside_run_ignored: token 40 is the fourth draft of round 2, and the
    # bonus token after it must not be emitted.
    decode(ctx, 3, 0.9, PROMPT, 4, 12, 40, expected=CONTINUATION[:5])


@case("a draft that mostly agrees: 40 tokens in fewer than 20 target calls", public=True)
def high_agreement(ctx):
    # Rejects one_draft_checked, drops_bonus_token, and both cache rebuilds.
    for seed, prompt, k in [(4, [0, 8, 8, 8], 4), (9, [0, 12, 40], 6)]:
        got, accepted = decode(ctx, seed, 0.9, prompt, k, 40, None)
        if len(accepted) >= len(got) / 2:
            raise RuntimeError("authoring error: this fixture needs too many rounds")


@case("a draft that is often wrong: both caches are rolled back round after round",
      public=True)
def low_agreement(ctx):
    # Rejects target_cache_not_rolled_back (wrong tokens) and
    # draft_keeps_rejected_tokens (right tokens, too many rounds).
    _, accepted = decode(ctx, 5, 0.6, [0, 33, 2, 60, 14, 7], 3, 40, None)
    if not (accepted.count(0) >= 5 and any(0 < n < 3 for n in accepted)):
        raise RuntimeError("authoring error: this fixture no longer mixes rejections")


@case("max_new_tokens falls inside a block: the result has exactly max_new_tokens tokens")
def budget_inside_block(ctx):
    # Rejects overshoots_max_new_tokens: with a full block the second round would
    # yield 5 tokens where only 2 are left.
    for limit in (7, 6, 2, 9):
        got, _ = decode(ctx, 4, 0.9, [0, 8, 8, 8], 4, limit, None)
        if len(got) != limit:
            raise RuntimeError("authoring error: generation stopped early")


@case("max_new_tokens = 1: one target call, and nothing is drafted for a token nobody needs")
def single_new_token(ctx):
    decode(ctx, 3, 0.9, PROMPT, 4, 1, None)
    decode(ctx, 9, 0.9, [0, 12, 40], 6, 1, 4)


@case("EOS as a draft token, as the bonus token, as the correction, and as the first token")
def eos_positions(ctx):
    # Seed 9 accepts all six drafts of round 1: [4, 55, 40, 3, 62, 13], bonus 26.
    for eos_id, length in [(55, 2), (3, 4), (13, 6), (26, 7)]:
        got, accepted = decode(ctx, 9, 0.9, [0, 12, 40], 6, 40, eos_id)
        if len(got) != length or accepted != [6]:
            raise RuntimeError("authoring error: EOS is not where this case expects it")
    # Seed 11 rejects the first draft, and the target's correction is EOS.
    got, accepted = decode(ctx, 11, 0.9, [0, 9], 4, 30, 1)
    if got != [1] or accepted != [0]:
        raise RuntimeError("authoring error: this model no longer starts with EOS")
    # EOS reached several rounds in.
    decode(ctx, 4, 0.9, [0, 8, 8, 8], 4, 40, 3)


@case("an eos_id that occurs only in the prompt does not stop generation")
def eos_in_prompt(ctx):
    got, _ = decode(ctx, 3, 0.9, PROMPT, 4, 8, 17)
    if len(got) != 8:
        raise RuntimeError("authoring error: token 17 is generated in this case")


@case("draft lengths k = 1 and k = 8")
def draft_lengths(ctx):
    decode(ctx, 3, 0.9, PROMPT, 1, 20, None)
    decode(ctx, 4, 0.9, [0, 8, 8, 8], 8, 45, None)


@case("a draft identical to the target: every block is accepted in full")
def perfect_draft(ctx):
    _, accepted = decode(ctx, 6, 1.0, [0, 5, 6], 4, 23, None)
    if accepted != [4, 4, 4, 4, 2]:
        raise RuntimeError("authoring error: a perfect draft was rejected")
    decode(ctx, 11, 0.9, [0], 3, 20, None)


@case("a 40-token prompt and 80 new tokens with k = 8: the whole context window is used")
def long_prompt(ctx):
    # P + max_new_tokens + k is exactly max_context. Each prompt token may pass
    # through each model once: the limits leave no room for feeding it again.
    decode(ctx, 9, 0.9, [0, *range(20, 59)], 8, 80, None)

