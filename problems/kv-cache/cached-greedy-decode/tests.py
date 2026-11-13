"""Tests for cached-greedy-decode.

Every case checks the generated tokens first, against greedy decoding done here
without a cache (the whole sequence through the model at every step), and then the
cost through the model's counters. For a generation of `g` tokens from a prompt of
`P` tokens the limits are `g` forward calls and `P + g - 1` tokens processed: the
prompt once, in one call, then each generated token except the last, one per call.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import TinyLM

FN = "greedy_decode"

# The worked example of the statement: TinyLM(seed=3) continues this prompt with
# 42, 62, 33, 33, 40, 32, 45, 37.
PROMPT = [0, 17, 42, 5]
CONTINUATION = [42, 62, 33, 33, 40, 32, 45, 37]


def uncached_greedy(ctx, seed, prompt, max_new_tokens, eos_id):
    """The definition: greedy decoding that re-runs the full sequence at every step."""
    lm = TinyLM(ctx.oracle_fw, seed=seed)
    sequence, generated = list(prompt), []
    while len(generated) < max_new_tokens:
        logits, _ = lm(sequence)
        token = int(np.argmax(ctx.oracle_fw.numpy(logits)[-1]))
        generated.append(token)
        sequence.append(token)
        if token == eos_id:
            break
    return generated


def decode(ctx, seed, prompt, max_new_tokens, eos_id, expected=None):
    """Run the solution; check its tokens, then its cost. Returns the tokens."""
    if expected is None:
        expected = uncached_greedy(ctx, seed, prompt, max_new_tokens, eos_id)
    lm = TinyLM(ctx.fw, seed=seed)
    got = ctx.call(FN, lm, list(prompt), max_new_tokens, eos_id)
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{FN}: must return a list of Python ints",
        f"got {got!r}",
    )
    check.equal(got, expected, label="generated tokens")
    produced = len(expected)
    check.cost(lm.counters, "forward_calls", at_most=produced)
    check.cost(lm.counters, "tokens_processed", at_most=len(prompt) + produced - 1, unit="tokens")
    return got


@case("the worked example: 8 tokens from a 4-token prompt, 8 calls and 11 tokens", public=True)
def worked_example(ctx):
    decode(ctx, 3, PROMPT, 8, 1, expected=CONTINUATION)


@case("stops at EOS: with eos_id = 40 the worked example ends after 5 tokens", public=True)
def stops_at_eos(ctx):
    # 5 calls and 4 + 4 = 8 tokens: the EOS token is returned but never fed.
    decode(ctx, 3, PROMPT, 8, 40, expected=CONTINUATION[:5])


@case("a 24-token generation equals uncached greedy decoding, at linear cost", public=True)
def matches_uncached(ctx):
    decode(ctx, 3, PROMPT, 24, None)


@case("another model: 60 tokens from a 6-token prompt")
def long_generation(ctx):
    # Re-feeding the prefix here costs 2130 tokens instead of 65.
    decode(ctx, 5, [0, 33, 2, 60, 14, 7], 60, None)


@case("a 40-token prompt and 3 new tokens: the prompt goes in as one call")
def long_prompt(ctx):
    # Catches a prompt fed token by token (42 calls instead of 3).
    decode(ctx, 7, [0, *range(20, 59)], 3, 1)


@case("EOS as the very first generated token: one call, nothing after it")
def eos_first(ctx):
    got = decode(ctx, 11, [0, 9], 40, 1)
    if got != [1]:
        raise RuntimeError("authoring error: this model no longer starts with EOS")


@case("max_new_tokens = 1: one call for the prompt, and the new token is not fed")
def single_new_token(ctx):
    decode(ctx, 3, PROMPT, 1, 1)
    decode(ctx, 9, [0, 12, 40], 1, None)


@case("EOS at different depths on three models, including at the last allowed token")
def eos_depths(ctx):
    # The stop token is chosen from what each model really generates, so EOS is
    # reached at step 7, at step 15, and exactly at step max_new_tokens.
    for seed, prompt, limit, depth in [(2, [0, 50, 51], 40, 6), (4, [0, 8, 8, 8], 40, 14)]:
        free = uncached_greedy(ctx, seed, prompt, limit, None)
        eos_id = free[depth]
        stop = free.index(eos_id)
        got = decode(ctx, seed, prompt, limit, eos_id)
        if len(got) != stop + 1 or len(got) == limit:
            raise RuntimeError("authoring error: EOS is not reached early in this case")
    free = uncached_greedy(ctx, 9, [0, 12, 40], 12, None)
    if free.index(free[-1]) != 11:
        raise RuntimeError("authoring error: the last token occurs earlier")
    decode(ctx, 9, [0, 12, 40], 12, free[-1])


@case("an eos_id that occurs only in the prompt does not stop generation")
def eos_in_prompt(ctx):
    got = decode(ctx, 3, PROMPT, 8, 17)
    if len(got) != 8:
        raise RuntimeError("authoring error: token 17 is generated in this case")


@case("a single-token prompt")
def single_token_prompt(ctx):
    decode(ctx, 11, [0], 20, None)
