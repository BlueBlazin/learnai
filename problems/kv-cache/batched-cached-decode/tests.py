"""Tests for batched-cached-decode.

Every case checks three things, in this order:

1. the tokens of every row, against greedy decoding done here one row at a time and
   without a cache (the definition);
2. the shape of the work: every call carries all B rows, the prompts go in as one
   call, each later call is one token per row, and a finished row is fed `eos_id`;
3. the cost: with `g_max` the length of the longest returned row, exactly `g_max`
   forward calls and `B * (P + g_max - 1)` tokens processed.

The model handed to the solution is a `TinyLM` that also writes down what it was
called with (`RecordingLM`), which is how point 2 is checked.
"""

import numpy as np

from whetstone.judge import case, check
from whetstone.kit import TinyLM

FN = "batched_greedy"

# The worked example of the statement, on TinyLM(seed=3) with eos_id = 18.
PROMPTS = [[0, 52, 7, 13], [0, 16, 13, 51], [0, 55, 38, 4]]
ROWS = [[24, 7, 51, 16, 16, 18], [16, 18], [5, 60, 26, 41, 37, 62, 30, 7, 18]]


class RecordingLM(TinyLM):
    """A TinyLM that records the tokens of every call, as numpy arrays."""

    def __init__(self, fw, seed):
        super().__init__(fw, seed=seed)
        self.calls = []

    def __call__(self, tokens, cache=None):
        self.calls.append(np.array(self._token_array(tokens)))
        return super().__call__(tokens, cache)


def uncached_greedy(ctx, seed, prompt, max_new_tokens, eos_id):
    """The definition, for one row: re-run the full sequence at every step."""
    lm = TinyLM(ctx.oracle_fw, seed=seed)
    sequence, generated = list(prompt), []
    while len(generated) < max_new_tokens:
        logits = ctx.oracle_fw.numpy(lm(sequence)[0])[-1]
        top = np.sort(logits)[-2:]
        if top[1] - top[0] < 1e-3:  # the argmax must not depend on the call pattern
            raise RuntimeError("authoring error: the two largest logits are too close")
        token = int(np.argmax(logits))
        generated.append(token)
        sequence.append(token)
        if token == eos_id:
            break
    return generated


def decode(ctx, seed, prompts, max_new_tokens, eos_id, expected=None):
    """Run the solution; check its tokens, then its calls, then its cost."""
    if expected is None:
        expected = [uncached_greedy(ctx, seed, p, max_new_tokens, eos_id) for p in prompts]
    lm = RecordingLM(ctx.fw, seed)
    got = ctx.call(FN, lm, [list(p) for p in prompts], max_new_tokens, eos_id)
    check.that(
        isinstance(got, list)
        and all(isinstance(row, list) and all(type(t) is int for t in row) for row in got),
        f"{FN}: must return a list with one list of Python ints per row",
        f"got {got!r}",
    )
    check.equal(got, expected, label="generated tokens")

    n_rows, n_prompt = len(prompts), len(prompts[0])
    longest = max(len(row) for row in expected)
    shapes = [call.shape for call in lm.calls]
    detail = f"shapes of the token arrays passed to lm, in call order: {shapes}"
    check.that(
        all(len(shape) == 2 and shape[0] == n_rows for shape in shapes),
        f"{FN}: every call to lm must carry all B rows (tokens of shape (B, T))",
        detail,
    )
    check.that(
        shapes[0] == (n_rows, n_prompt) and all(shape[1] == 1 for shape in shapes[1:]),
        f"{FN}: the first call must pass the whole prompts, every later call one token per row",
        detail,
    )
    # Call k + 2 feeds, per row, generated token k, or eos_id once the row has finished.
    # Calls beyond the longest row are left to the cost check below.
    for k, call in enumerate(lm.calls[1:longest]):
        want = [row[k] if k < len(row) and row[k] != eos_id else eos_id for row in expected]
        check.that(
            call[:, 0].tolist() == want,
            f"{FN}: a decode call was passed the wrong tokens (a running row gets its newest "
            "token, a finished row gets eos_id)",
            f"call {k + 2}: got {call[:, 0].tolist()}, expected {want}",
        )
    check.cost(lm.counters, "forward_calls", at_most=longest)
    check.cost(
        lm.counters, "tokens_processed", at_most=n_rows * (n_prompt + longest - 1), unit="tokens"
    )
    return got


@case("the worked example: three rows finish after 6, 2, and 9 tokens; 9 calls", public=True)
def worked_example(ctx):
    decode(ctx, 3, PROMPTS, 12, 18, expected=ROWS)


@case("no row emits EOS: every row runs to max_new_tokens", public=True)
def no_eos(ctx):
    got = decode(ctx, 3, PROMPTS, 5, 1)
    if any(len(row) != 5 for row in got):
        raise RuntimeError("authoring error: a row emits EOS in this case")


@case("one row hits the budget while the others end with EOS", public=True)
def budget_and_eos(ctx):
    # Rows 0 and 1 end with EOS (18) after 6 and 2 tokens; row 2 would need 9 and is
    # cut at 7 without an EOS.
    decode(ctx, 3, PROMPTS, 7, 18, expected=[ROWS[0], ROWS[1], ROWS[2][:7]])


@case("all rows finish long before max_new_tokens: no call after the last EOS")
def all_finish_early(ctx):
    # Catches a loop that always runs max_new_tokens steps (40 calls instead of 9), and
    # one that notices "all finished" only after another forward call.
    decode(ctx, 3, PROMPTS, 40, 18)


@case("another model, five rows: EOS in the first step for one row, never for another")
def five_rows(ctx):
    prompts = [[0, 43, 51, 3], [0, 52, 31, 33], [0, 41, 19, 62], [0, 5, 19, 25], [0, 37, 27, 10]]
    got = decode(ctx, 5, prompts, 12, 19)
    lengths = [len(row) for row in got]
    if lengths != [12, 12, 1, 12, 2]:
        raise RuntimeError(f"authoring error: the row lengths changed to {lengths}")


@case("a row that would emit EOS again after its end, and rows that finish together")
def repeated_eos(ctx):
    # On this model rows 2 and 3 generate EOS (token 1), keep generating it when fed it,
    # and row 4 reaches it at the very last allowed token.
    prompts = [[0, 60, 40, 44], [0, 57, 37, 50], [0, 53, 15, 5], [0, 20, 19, 56], [0, 58, 2, 32]]
    got = decode(ctx, 7, prompts, 14, 1)
    lengths = [len(row) for row in got]
    if lengths != [14, 14, 5, 4, 14] or got[4][-1] != 1:
        raise RuntimeError(f"authoring error: the row lengths changed to {lengths}")


@case("eos_id = None: exactly max_new_tokens tokens per row")
def eos_none(ctx):
    decode(ctx, 5, [[0, 43, 51, 3], [0, 41, 19, 62]], 9, None)


@case("max_new_tokens = 1: the prefill is the only call")
def single_new_token(ctx):
    decode(ctx, 3, PROMPTS, 1, 18)
    decode(ctx, 7, [[0, 53, 15, 5], [0, 20, 19, 56]], 1, None)


@case("a batch of one row, and a batch whose rows are identical")
def one_row_and_identical_rows(ctx):
    decode(ctx, 3, [PROMPTS[2]], 20, 18)
    decode(ctx, 3, [PROMPTS[0], PROMPTS[0], PROMPTS[0]], 20, 18)


@case("an eos_id that occurs only in the prompts stops nothing; 30-token prompts")
def long_prompts(ctx):
    prompts = [[0, *range(20 + b, 49 + b)] for b in range(4)]
    got = decode(ctx, 7, prompts, 4, 25)
    if any(len(row) != 4 for row in got):
        raise RuntimeError("authoring error: token 25 is generated in this case")
