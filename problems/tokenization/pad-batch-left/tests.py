"""Tests for pad-batch-left.

The worked examples are compared with literal values, everything else with the oracle.
Every comparison goes through `run`, which also checks the structure of the result
(a 3-tuple of lists of plain ints), so a wrong type is reported as such.
"""

import copy

from whetstone.judge import case, check


def is_int(value):
    return type(value) is int


def run(ctx, seqs, pad_id, *side):
    """Call the solution, check the shape and types of the result, and return it."""
    out = ctx.call("pad_batch", seqs, pad_id, *side)
    detail = f"got {out!r}"
    check.that(
        isinstance(out, tuple) and len(out) == 3,
        "pad_batch must return a tuple (ids, attention_mask, lengths)",
        detail,
    )
    ids, mask, lengths = out
    for name, grid in (("ids", ids), ("attention_mask", mask)):
        check.that(
            isinstance(grid, list)
            and all(isinstance(row, list) and all(is_int(v) for v in row) for row in grid),
            f"{name} must be a list of lists of ints",
            detail,
        )
    check.that(
        isinstance(lengths, list) and all(is_int(v) for v in lengths),
        "lengths must be a list of ints",
        detail,
    )
    return ids, mask, lengths


def expect(ctx, seqs, pad_id, side, want):
    """Compare with literal values. `side` is a tuple: () calls with the default."""
    ids, mask, lengths = run(ctx, seqs, pad_id, *side)
    check.equal(ids, want[0], label="ids")
    check.equal(mask, want[1], label="attention_mask")
    check.equal(lengths, want[2], label="lengths")


def same_as_oracle(ctx, seqs, pad_id, *side):
    expect(ctx, seqs, pad_id, side, ctx.expected("pad_batch", seqs, pad_id, *side))


@case("the worked example: three prompts, padded on the left by default", public=True)
def worked_example(ctx):
    # Rejects sides_swapped, mask_inverted, lengths_are_width, default_right.
    seqs = [[5, 6, 7], [8], [9, 10]]
    want = (
        [[5, 6, 7], [0, 0, 8], [0, 9, 10]],
        [[1, 1, 1], [0, 0, 1], [0, 1, 1]],
        [3, 1, 2],
    )
    expect(ctx, seqs, 0, (), want)
    expect(ctx, seqs, 0, ("left",), want)


@case('the same batch with side="right"', public=True)
def right_side(ctx):
    seqs = [[5, 6, 7], [8], [9, 10]]
    want = (
        [[5, 6, 7], [8, 0, 0], [9, 10, 0]],
        [[1, 1, 1], [1, 0, 0], [1, 1, 0]],
        [3, 1, 2],
    )
    expect(ctx, seqs, 0, ("right",), want)


@case("a real token equal to pad_id stays visible in the mask", public=True)
def pad_id_inside_sequence(ctx):
    # Rejects mask_from_pad_id: here the pad token is also the end-of-sequence token.
    seqs = [[7, 2, 9, 2], [2], [4, 2]]
    want_left = (
        [[7, 2, 9, 2], [2, 2, 2, 2], [2, 2, 4, 2]],
        [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]],
        [4, 1, 2],
    )
    expect(ctx, seqs, 2, ("left",), want_left)
    same_as_oracle(ctx, seqs, 2, "right")
    same_as_oracle(ctx, [[2, 2], [2, 2, 2, 2], [2]], 2)


@case("an empty sequence, an empty batch, and an unknown side raise ValueError", public=True)
def errors(ctx):
    # Rejects empty_row_allowed and unknown_side_is_right.
    fn = ctx.solution.pad_batch
    check.raises(ValueError, fn, [[1, 2], [], [3]], 0, label="pad_batch([[1, 2], [], [3]], 0)")
    check.raises(ValueError, fn, [[]], 0, "right", label='pad_batch([[]], 0, "right")')
    check.raises(ValueError, fn, [], 0, label="pad_batch([], 0)")
    check.raises(ValueError, fn, [[1], [2, 3]], 0, "center", label='side="center"')
    check.raises(ValueError, fn, [[1], [2, 3]], 0, "Left", label='side="Left"')
    check.raises(ValueError, fn, [[1, 2]], 0, "", label='side=""')


@case("the default side is left")
def default_is_left(ctx):
    for seqs in ([[1], [2, 3, 4]], [[4, 5, 6, 7, 8], [9]]):
        ids, mask, lengths = run(ctx, seqs, 63)
        want = ctx.expected("pad_batch", seqs, 63, "left")
        check.equal(ids, want[0], label="ids with the default side")
        check.equal(mask, want[1], label="attention_mask with the default side")
        check.equal(lengths, want[2], label="lengths with the default side")


@case("an unknown side is an error even when no row needs padding")
def side_validated_always(ctx):
    fn = ctx.solution.pad_batch
    check.raises(ValueError, fn, [[1, 2], [3, 4]], 0, "both", label='side="both"')
    check.raises(ValueError, fn, [[1]], 0, "RIGHT", label='side="RIGHT"')
    check.raises(ValueError, fn, [[1], [2, 3]], 0, "l", label='side="l"')


@case("an empty sequence is an error wherever it is, on either side")
def empty_rows(ctx):
    fn = ctx.solution.pad_batch
    for seqs in ([[], [1]], [[1], []], [[1, 2, 3], [4], [], [5, 6]], [[], []]):
        for side in ("left", "right"):
            check.raises(ValueError, fn, copy.deepcopy(seqs), 9, side, label="an empty sequence")


@case("a single sequence, equal lengths, and sequences of length one")
def nothing_to_pad(ctx):
    for seqs in ([[3, 1, 4, 1, 5]], [[1, 2], [3, 4], [5, 6]], [[7], [8], [9]], [[0]]):
        for side in ("left", "right"):
            same_as_oracle(ctx, seqs, 0, side)


@case("the input is not modified and the result does not share lists with it")
def input_untouched(ctx):
    # Rejects pads_in_place. Called directly so the solution sees these very lists.
    fn = ctx.solution.pad_batch
    for side in ("left", "right"):
        seqs = [[5, 6, 7], [8], [9, 10], [1, 2, 3]]
        before = copy.deepcopy(seqs)
        ids, _, _ = fn(seqs, 0, side)
        check.that(seqs == before, "pad_batch modified its input", f"side={side!r}: {seqs!r}")
        check.that(
            all(row is not seq for row in ids for seq in seqs),
            "a row of ids is the same list object as an input sequence",
            "rows of the result must be new lists, also for rows that need no padding",
        )
        again = fn(seqs, 0, side)
        check.equal(list(again[0]), list(ids), label="ids of a second call on the same input")


@case("random batches on both sides, with pad ids that also occur as tokens")
def random_batches(ctx):
    for _ in range(40):
        batch = int(ctx.rng.integers(1, 9))
        seqs = [
            [int(t) for t in ctx.rng.integers(0, 12, size=int(ctx.rng.integers(1, 14)))]
            for _ in range(batch)
        ]
        pad_id = int(ctx.rng.integers(0, 12))
        same_as_oracle(ctx, seqs, pad_id, "left")
        same_as_oracle(ctx, seqs, pad_id, "right")


@case("a wide batch: one long prompt among short ones")
def wide_batch(ctx):
    seqs = [[int(t) for t in ctx.rng.integers(3, 50000, size=n)] for n in (1, 512, 3, 77, 1, 511)]
    same_as_oracle(ctx, seqs, 50256, "left")
    same_as_oracle(ctx, seqs, 50256, "right")
