"""Tests for greedy-speculative-verify.

Every case is exact. The expected pair is computed here, in numpy, straight from the
definition (the first position where the target's argmax differs from the draft), and
the inputs are built so that no row has a near-tie: `tie_free` checks that the two
largest logits of every row are at least 1e-3 apart, as the statement promises.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import draft_target_pair

FN = "verify_greedy"

# The worked example of the statement. The target's greedy tokens are 2, 0, 3, 1.
LOGITS = np.array(
    [
        [0.1, -1.0, 2.5, 0.3, 0.0],
        [1.8, 0.2, -0.5, 1.1, 0.4],
        [0.0, 0.5, -1.2, 2.0, 1.5],
        [-0.3, 2.2, 0.1, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def tie_free(logits):
    top = np.sort(logits.astype(np.float64), axis=-1)[:, -2:]
    return bool(np.all(top[:, 1] - top[:, 0] >= 1e-3))


def definition(draft, logits):
    """(n_accepted, next_token) by the definition: stop at the first mismatch."""
    greedy = logits.argmax(axis=-1)
    n = 0
    while n < len(draft) and greedy[n] == draft[n]:
        n += 1
    return n, int(greedy[n])


def verify(ctx, draft, logits, expected=None):
    draft = np.asarray(draft, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape[0] != len(draft) + 1 or not tie_free(logits):  # an authoring error
        raise RuntimeError("bad test input: wrong number of rows, or a near-tie in a row")
    result = ctx.call(FN, draft, logits)
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{FN}: expected an (n_accepted, next_token) pair, got {type(result).__name__}",
    )
    values = [np.asarray(value) for value in result]
    check.that(
        all(value.size == 1 and value.dtype.kind in "iu" for value in values),
        f"{FN}: n_accepted and next_token must each be a single integer",
        f"got {result!r}",
    )
    got = tuple(int(value.reshape(())) for value in values)
    want = definition(draft, logits) if expected is None else expected
    detail = (
        f"call: {FN}(draft_tokens, target_logits)\n  draft_tokens = {draft.tolist()}\n"
        f"  argmax of each row of target_logits = {logits.argmax(axis=-1).tolist()}\n"
        f"expected (n_accepted, next_token) = {want}\ngot {got}"
    )
    check.that(got[0] == want[0], f"{FN}: wrong number of accepted tokens", detail)
    check.that(got[1] == want[1], f"{FN}: wrong next token", detail)
    return got


def logits_for(ctx, greedy, vocab, scale=2.0):
    """Random float32 logits whose row-wise argmax is `greedy`, with a clear margin."""
    logits = ctx.randn(len(greedy), vocab, scale=scale)
    logits[np.arange(len(greedy)), greedy] = logits.max(axis=-1) + 0.5
    return logits


@case("the worked example: a mismatch at position 2, a full match, a mismatch at 0", public=True)
def worked_example(ctx):
    verify(ctx, [2, 0, 4], LOGITS, expected=(2, 3))
    verify(ctx, [2, 0, 3], LOGITS, expected=(3, 1))
    verify(ctx, [1, 0, 3], LOGITS, expected=(0, 2))


@case("every draft token matches: all K are accepted and the last row is the bonus", public=True)
def full_match(ctx):
    # Rejects no_bonus_token: the bonus differs from the last draft token.
    greedy = np.array([7, 3, 3, 9, 1, 4, 11])
    verify(ctx, greedy[:-1], logits_for(ctx, greedy, 12))


@case("matches after the first mismatch do not count", public=True)
def later_matches(ctx):
    # Rejects counts_later_matches and next_from_last_row.
    greedy = np.array([5, 2, 8, 6, 1, 0, 3])
    draft = greedy[:-1].copy()
    draft[2] = 4  # positions 3, 4, 5 still equal the target's argmax
    verify(ctx, draft, logits_for(ctx, greedy, 10))
    draft[0] = 9  # now nothing is accepted
    verify(ctx, draft, logits_for(ctx, greedy, 10))


@case("the first mismatch at every position of a block of 6", public=True)
def every_mismatch_position(ctx):
    # Rejects all_or_nothing (a matching prefix must be kept) and next_from_last_row.
    greedy = np.array([10, 4, 13, 2, 7, 0, 5])
    logits = logits_for(ctx, greedy, 16)
    for position in range(6):
        draft = greedy[:-1].copy()
        draft[position] = (draft[position] + 1 + position) % 16
        verify(ctx, draft, logits)


@case("a draft that equals the target's argmax one row later is not a match")
def shifted_draft(ctx):
    # Rejects shifted_alignment: the draft agrees with rows 1..K, not with rows 0..K-1.
    greedy = np.array([3, 8, 1, 6, 12, 9])
    logits = logits_for(ctx, greedy, 14)
    verify(ctx, greedy[1:], logits)
    # And the other way round: aligned with rows 0..K-1, while rows 1..K all differ.
    verify(ctx, greedy[:-1], logits)


@case("K = 1: one draft token, accepted or rejected")
def single_draft(ctx):
    greedy = np.array([4, 2])
    logits = logits_for(ctx, greedy, 6)
    verify(ctx, [4], logits)
    verify(ctx, [2], logits)
    verify(ctx, [0], logits)


@case("masked logits: rows that contain -inf, and a square (K + 1) x V block")
def masked_rows(ctx):
    greedy = np.array([2, 0, 3, 3, 1])
    logits = logits_for(ctx, greedy, 5)  # 5 rows, 5 columns
    logits[0, [0, 4]] = -np.inf
    logits[2, [1]] = -np.inf
    logits[4, [0, 2, 3, 4]] = -np.inf  # one finite entry
    verify(ctx, [2, 0, 3, 3], logits)
    verify(ctx, [2, 0, 1, 3], logits)


@case("negative logits only, and a repeated token: the target emits the same token every time")
def negative_and_repeated(ctx):
    greedy = np.full(9, 6)
    logits = logits_for(ctx, greedy, 8) - 40.0
    verify(ctx, np.full(8, 6), logits)
    draft = np.full(8, 6)
    draft[5] = 7
    verify(ctx, draft, logits)


@case("logits from a real draft and target model pair, verified round by round")
def model_logits(ctx):
    draft_lm, target_lm = draft_target_pair(ctx.oracle_fw, seed=3, agreement=0.8)
    as_numpy = ctx.oracle_fw.numpy
    sequence, accepted = [0, 17, 42, 5], []
    for _ in range(5):
        drafts = []
        for _ in range(4):
            logits, _ = draft_lm(sequence + drafts)
            drafts.append(int(as_numpy(logits)[-1].argmax()))
        logits, _ = target_lm(sequence + drafts)
        rows = as_numpy(logits)[len(sequence) - 1 :]
        n, token = verify(ctx, drafts, rows)
        accepted.append(n)
        sequence += [*drafts[:n], token]
    if len(set(accepted)) < 2:  # an authoring error
        raise RuntimeError("the model case no longer mixes acceptances and rejections")


@case("random blocks: K from 1 to 16, V from 2 to 64, mismatches anywhere or nowhere")
def random_blocks(ctx):
    for k, vocab in [(1, 2), (3, 64), (16, 64), (8, 3), (16, 17), (5, 6), (12, 64), (2, 9)]:
        greedy = ctx.randint(0, vocab, k + 1)
        logits = logits_for(ctx, greedy, vocab, scale=float(10.0 ** ctx.rng.uniform(-1, 2)))
        draft = greedy[:-1].copy()
        wrong = ctx.rng.random(k) < 0.25
        draft[wrong] = (draft[wrong] + ctx.randint(1, vocab, int(wrong.sum()))) % vocab
        verify(ctx, draft, logits)
