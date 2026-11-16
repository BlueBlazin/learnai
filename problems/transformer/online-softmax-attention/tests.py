"""Tests for online-softmax-attention.

The expected output is ordinary attention over all keys at once, computed in float64
by `dense`. Every case goes through `attend`, which also checks that all blocks were
read and that the solution let go of each block before asking for the next but one.

`BlockStream` is the one-pass iterator. It creates the tensors of a block when the
block is requested and keeps only weak references to them, so it can tell which
earlier blocks the solution still holds. `peak_blocks_held` is the largest number of
earlier blocks alive at a request (including the request that ends the iteration).
A loop `for k, v, mask in kv_blocks` holds exactly one: the block it just processed.

Most cases use D = 4, so the score scale 1 / sqrt(D) = 0.5 is exact; cases with
scores in the hundreds use integer-valued queries and keys, for which the scores are
exact in float32 whatever the order of evaluation.
"""

import weakref
from itertools import pairwise

import numpy as np

from learnai.judge import case, check

FN = "blockwise_attention"
TOL = {"rtol": 1e-4, "atol": 1e-5}  # the tolerance stated in Constraints
T, D, DV, N = 5, 4, 6, 16
PARTITIONS = [[16], [8, 8], [4, 4, 4, 4], [2] * 8, [8, 4, 2, 2], [2, 2, 4, 8], [1, 1, 2, 4, 8]]


class BlockStream:
    """A one-pass iterator of (k, v, mask) blocks that knows which ones are still alive."""

    def __init__(self, fw, blocks):
        self.fw = fw
        self._blocks = blocks
        self._next = 0
        self._live = []
        self.peak_held = 0

    def __iter__(self):
        return self

    def __next__(self):
        held = sum(any(ref() is not None for ref in refs) for refs in self._live)
        self.peak_held = max(self.peak_held, held)
        if self._next == len(self._blocks):
            raise StopIteration
        tensors = tuple(self.fw.tensor(array) for array in self._blocks[self._next])
        self._next += 1
        self._live.append(tuple(weakref.ref(tensor) for tensor in tensors))
        return tensors

    @property
    def blocks_read(self):
        return self._next


def split(k, v, mask, sizes):
    """Cut keys, values, and mask columns into consecutive blocks of the given sizes."""
    edges = np.cumsum([0, *sizes])
    if edges[-1] != len(k):
        raise RuntimeError("authoring error: block sizes do not add up to the number of keys")
    return [(k[a:b].copy(), v[a:b].copy(), mask[:, a:b].copy()) for a, b in pairwise(edges)]


def dense(q, blocks):
    """Attention over all keys at once, in float64."""
    k = np.concatenate([b[0] for b in blocks]).astype(np.float64)
    v = np.concatenate([b[1] for b in blocks]).astype(np.float64)
    mask = np.concatenate([b[2] for b in blocks], axis=1)
    if not mask.any(axis=1).all():
        raise RuntimeError("authoring error: a query sees no key")
    scores = np.where(mask, q.astype(np.float64) @ k.T / np.sqrt(q.shape[1]), -np.inf)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return ((weights / weights.sum(axis=1, keepdims=True)) @ v).astype(np.float32)


def attend(ctx, q, blocks):
    """Run the solution over a stream of blocks; check the output and what it held."""
    stream = BlockStream(ctx.fw, blocks)
    out = ctx.call(FN, q, stream)
    check.that(
        stream.blocks_read == len(blocks),
        f"{FN}: returned after reading {stream.blocks_read} blocks, before the iterator ended",
    )
    check.allclose(out, dense(q, blocks), **TOL, label=FN, dtype=True)
    check.at_most("peak_blocks_held", stream.peak_held, 1, unit="blocks")
    return out


def random_inputs(ctx, t=T, d=D, dv=DV, n=N, scale=1.0):
    q = ctx.randn(t, d, scale=scale)
    return q, ctx.randn(n, d), ctx.randn(n, dv), np.ones((t, n), dtype=bool)


def integer_inputs(ctx, q_range, t=T, n=N):
    """Integer-valued queries and keys: scores are exact, and reach +-q_range * 20 / 2."""
    q = ctx.rng.integers(-q_range, q_range + 1, (t, D)).astype(np.float32)
    k = ctx.rng.integers(-5, 6, (n, D)).astype(np.float32)
    return q, k, ctx.randn(n, DV), np.ones((t, n), dtype=bool)


@case("the worked example: three keys in two blocks, a score of 100, a masked block", public=True)
def worked_example(ctx):
    q = np.array([[1.0], [100.0]], dtype=np.float32)
    blocks = [
        (
            np.array([[0.0], [1.0]], dtype=np.float32),
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[True, True], [True, True]]),
        ),
        (
            np.array([[3.0]], dtype=np.float32),
            np.array([[4.0]], dtype=np.float32),
            np.array([[True], [False]]),
        ),
    ]
    out = attend(ctx, q, blocks)
    check.allclose(out, np.array([[3.645579], [2.0]], dtype=np.float32), **TOL, label=FN)


@case("random keys in four blocks of four: the result is attention over all sixteen", public=True)
def four_blocks(ctx):
    # Catches a normalizer per block, and sums that are not rescaled when the maximum grows.
    q, k, v, mask = random_inputs(ctx, scale=3.0)
    attend(ctx, q, split(k, v, mask, [4, 4, 4, 4]))


@case(
    "scores from -300 to 300, with each query's best key in a different block",
    public=True,
)
def large_scores(ctx):
    # exp(300) overflows float32 and exp(-300) underflows. Queries point in different
    # directions, so one query's maximum is hundreds above another's. Catches a plain
    # running sum of exponentials and a maximum shared between queries.
    q, k, v, mask = integer_inputs(ctx, 30)
    for sizes in ([4, 4, 4, 4], [2] * 8):
        attend(ctx, q, split(k, v, mask, sizes))
    # The same keys sorted by the first query's score, ascending (its maximum grows
    # with every block) and descending (the first block already holds its maximum).
    order = np.argsort(k @ q[0], kind="stable")
    for index in (order, order[::-1]):
        attend(ctx, q, split(k[index], v[index], mask, [4, 4, 4, 4]))


@case(
    "masked blocks: a block no query may attend to, and queries whose first blocks are masked",
    public=True,
)
def masked_blocks(ctx):
    # A query that has seen no visible key has a running maximum of -inf. Catches
    # -inf - (-inf) = NaN, and a mask that is ignored.
    q, k, v, mask = random_inputs(ctx, scale=3.0)
    mask[:, 4:8] = False  # the second block: masked for everyone
    mask[1, :8] = False  # query 1: nothing visible in the first two blocks
    mask[2, :12] = False  # query 2: only the last block
    mask[3, 12:] = False
    mask[4, ::2] = False
    attend(ctx, q, split(k, v, mask, [4, 4, 4, 4]))
    mask[:, :4] = False  # now the first block is masked for everyone
    mask[0, 12:] = False
    attend(ctx, q, split(k, v, mask, [4, 4, 4, 4]))


@case("one pass, one block at a time: eight blocks, and no earlier block is kept", public=True)
def one_block_at_a_time(ctx):
    q, k, v, mask = random_inputs(ctx, scale=2.0)
    mask[ctx.rng.random(mask.shape) < 0.25] = False
    mask[:, 0] = True
    attend(ctx, q, split(k, v, mask, [2] * 8))


@case("the result does not depend on how the keys are cut into blocks")
def partition_invariance(ctx):
    def prop(rng):
        q, k, v, mask = random_inputs(ctx, scale=float(10.0 ** rng.uniform(-0.5, 0.8)))
        mask[rng.random(mask.shape) < 0.3] = False
        mask[np.arange(T), rng.integers(N, size=T)] = True  # every query sees a key
        outs = [
            attend(ctx, q, split(k, v, mask, PARTITIONS[int(i)]))
            for i in rng.choice(len(PARTITIONS), size=3, replace=False)
        ]
        for other in outs[1:]:
            check.allclose(other, outs[0], **TOL, label="the same keys in other blocks")

    check.forall(prop, trials=4)


@case("a causal mask across blocks: later blocks are fully masked for early queries")
def causal_across_blocks(ctx):
    q, k, v, mask = random_inputs(ctx, scale=3.0)
    mask[:] = np.arange(N)[None, :] < 3 * (np.arange(T)[:, None] + 1)
    for sizes in ([4, 4, 4, 4], [8, 4, 2, 2], [1, 1, 2, 4, 8]):
        attend(ctx, q, split(k, v, mask, sizes))


@case("every score far below zero, and every score far above")
def all_scores_extreme(ctx):
    # Without a running maximum, exp underflows to 0 / 0 on the first and overflows to
    # inf / inf on the second.
    q, k, v, mask = integer_inputs(ctx, 3)
    k_low = k.copy()
    k_low[:, 0] = 5.0
    q_low, q_high = q.copy(), q.copy()
    q_low[:, 0] = -120.0  # adds -300 to every score
    q_high[:, 0] = 120.0  # adds +300 to every score
    for queries in (q_low, q_high):
        attend(ctx, queries, split(k_low, v, mask, [4, 4, 4, 4]))


@case("the same key in several blocks: the running maximum is met again, not exceeded")
def repeated_maximum(ctx):
    q, k, v, mask = integer_inputs(ctx, 8)
    k[4:8] = k[:4]
    k[12:] = k[:4]
    attend(ctx, q, split(k, v, mask, [4, 4, 4, 4]))


@case("a single block, blocks of one key, a single query, and D = 8 with one value column")
def shapes(ctx):
    q, k, v, mask = random_inputs(ctx, scale=2.0)
    attend(ctx, q, split(k, v, mask, [16]))
    attend(ctx, q[:1], split(k, v, mask[:1], [1, 1, 2, 4, 8]))
    q, k, v, mask = random_inputs(ctx, d=8, dv=1, scale=2.0)
    attend(ctx, q, split(k, v, mask, [8, 8]))
