"""Tests for batched-speculative-accept.

`verify_batch` is checked three ways:

- numeric: single calls against `definition`, a float64 loop over rows written from
  the statement of `speculative-sampling-block`. Inputs stay clear of every decision
  boundary (`decisive`), so float32 rounding can never decide a call;
- property: a batch gives, row by row, what the same function gives for that row
  alone (B = 1);
- statistical: the test plays both models for every row (draft token i of row b is
  drawn from q[b, i]; all uniforms independent) and compares, per row, the number of
  accepted drafts with its exact distribution and the first emitted token with
  p[b, 0], and, for the calls that accept every draft, the bonus token with p[b, K].

`advance_batch` is exact: its output is compared with `advance`, plain Python lists.

Statistical checks in this file: 12 at most, each with delta = 1e-9 / 12.
"""

import numpy as np

from whetstone.judge import case, check

VERIFY = "verify_batch"
ADVANCE = "advance_batch"
MARGIN = 1e-3
DELTA = 1e-9 / 12
SHAPES = [(1, 1, 1), (2, 5, 3), (3, 8, 4), (5, 6, 5), (4, 3, 2)]  # (B, T, K)

# The worked example of the statement.
P = np.array(
    [
        [[0.10, 0.40, 0.20, 0.30], [0.25, 0.25, 0.25, 0.25], [0.50, 0.10, 0.10, 0.30]],
        [[0.70, 0.10, 0.10, 0.10], [0.20, 0.20, 0.30, 0.30], [0.10, 0.20, 0.30, 0.40]],
    ],
    dtype=np.float32,
)
Q = np.array(
    [
        [[0.50, 0.20, 0.20, 0.10], [0.10, 0.60, 0.20, 0.10]],
        [[0.60, 0.20, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25]],
    ],
    dtype=np.float32,
)


def definition_row(p, q, drafts, u_accept, u_resample):
    """((n_accepted, next_token), decisive) for one row, in float64."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    decisive = True
    n = 0
    while n < len(drafts):
        ratio = p[n, drafts[n]] / q[n, drafts[n]]
        if 0 < ratio < 1 + MARGIN and abs(u_accept[n] - ratio) < MARGIN:
            decisive = False
        if not u_accept[n] * q[n, drafts[n]] < p[n, drafts[n]]:
            break
        n += 1
    weights = np.maximum(p[n] - q[n], 0.0) if n < len(drafts) else p[n]
    cdf = np.cumsum(weights)
    edges = cdf[cdf > 0] / cdf[-1]
    if np.abs(edges - u_resample).min() < MARGIN:
        decisive = False
    return (n, int(np.sum(cdf <= u_resample * cdf[-1]))), decisive


def definition(p, q, drafts, u_accept, u_resample):
    rows = [definition_row(*row) for row in zip(p, q, drafts, u_accept, u_resample, strict=True)]
    return [r[0][0] for r in rows], [r[0][1] for r in rows], all(r[1] for r in rows)


def int_vector(value, size, what):
    array = np.asarray(value)
    check.that(
        array.dtype.kind in "iu" and array.shape == (size,),
        f"{what}: must be an integer tensor with one entry per row",
        f"got dtype {array.dtype}, shape {array.shape}",
    )
    return [int(v) for v in array]


def verify(ctx, p, q, drafts, u_accept, u_resample):
    """Call the solution's verify_batch; return (n_accepted, next_tokens) as lists."""
    result = ctx.call(
        VERIFY,
        p,
        q,
        np.asarray(drafts, dtype=np.int64),
        np.asarray(u_accept, dtype=np.float32),
        np.asarray(u_resample, dtype=np.float32),
    )
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 2,
        f"{VERIFY}: expected an (n_accepted, next_tokens) pair, got {type(result).__name__}",
    )
    rows, k, vocab = q.shape
    n = int_vector(result[0], rows, f"{VERIFY}: n_accepted")
    tokens = int_vector(result[1], rows, f"{VERIFY}: next_tokens")
    check.that(all(0 <= v <= k for v in n), f"{VERIFY}: n_accepted is outside 0..K", f"got {n}")
    check.that(
        all(0 <= v < vocab for v in tokens), f"{VERIFY}: next_tokens holds a value that is "
        "not a token id", f"got {tokens}",
    )  # fmt: skip
    return n, tokens


def check_verify(ctx, p, q, drafts, u_accept, u_resample, literal=None):
    want_n, want_next, decisive = definition(p, q, drafts, u_accept, u_resample)
    if not decisive or (literal is not None and literal != (want_n, want_next)):
        raise RuntimeError("authoring error: ambiguous input, or a wrong literal")
    oracle = ctx.expected(
        VERIFY, p, q, np.asarray(drafts), np.asarray(u_accept, dtype=np.float32),
        np.asarray(u_resample, dtype=np.float32),
    )  # fmt: skip
    if [list(map(int, part)) for part in oracle] != [want_n, want_next]:
        raise RuntimeError("authoring error: the reference disagrees with the definition")
    got_n, got_next = verify(ctx, p, q, drafts, u_accept, u_resample)
    detail = (
        f"call: {VERIFY}(p, q, draft_tokens, u_accept, u_resample)\n  p =\n{p}\n  q =\n{q}\n"
        f"  draft_tokens = {np.asarray(drafts).tolist()}\n"
        f"  u_accept = {np.round(np.asarray(u_accept, dtype=np.float64), 4).tolist()}\n"
        f"  u_resample = {np.round(np.asarray(u_resample, dtype=np.float64), 4).tolist()}\n"
        f"expected n_accepted = {want_n}, next_tokens = {want_next}\n"
        f"got n_accepted = {got_n}, next_tokens = {got_next}"
    )
    wrong = [b for b in range(len(want_n)) if got_n[b] != want_n[b]]
    check.that(not wrong, f"{VERIFY}: wrong n_accepted in row {wrong[:1]}", detail)
    wrong = [b for b in range(len(want_n)) if got_next[b] != want_next[b]]
    check.that(not wrong, f"{VERIFY}: wrong next token in row {wrong[:1]}", detail)


def exact(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return rows / rows.sum(axis=-1, keepdims=True)


def models(ctx, rows, k, vocab, closeness):
    """p of shape (rows, k + 1, vocab); q blends p with another draw, per row."""
    closeness = np.broadcast_to(np.asarray(closeness, dtype=np.float64), (rows,))
    p = ctx.rng.dirichlet(np.full(vocab, 1.5), size=(rows, k + 1))
    other = ctx.rng.dirichlet(np.full(vocab, 1.5), size=(rows, k))
    q = closeness[:, None, None] * p[:, :k] + (1 - closeness[:, None, None]) * other
    return exact(p).astype(np.float32), exact(q).astype(np.float32)


def decisive_batch(ctx, p, q, tries=200):
    """Random drafts and uniforms for (p, q) that stay clear of every boundary."""
    rows, k, vocab = q.shape
    for _ in range(tries):
        drafts = ctx.randint(0, vocab, (rows, k))
        # Low uniforms in some rows, so that the batch mixes long and short prefixes.
        u_accept = ctx.rng.random((rows, k)) * ctx.rng.choice([0.3, 1.0], size=(rows, 1))
        u_resample = ctx.rng.random(rows)
        if definition(p, q, drafts, u_accept, u_resample)[2]:
            return drafts, u_accept, u_resample
    raise RuntimeError("authoring error: no decisive batch found")


def advance(tokens, lengths, drafts, n_accepted, next_tokens, pad_id):
    """The statement in plain Python: (rows as lists, new lengths, cache lengths)."""
    rows = [
        [*tokens[b][: lengths[b]], *drafts[b][: n_accepted[b]], next_tokens[b]]
        for b in range(len(lengths))
    ]
    width = max(len(row) for row in rows)
    matrix = [row + [pad_id] * (width - len(row)) for row in rows]
    return matrix, [len(row) for row in rows], [len(row) - 1 for row in rows]


def check_advance(ctx, tokens, lengths, drafts, n_accepted, next_tokens, pad_id, literal=None):
    args = [
        np.asarray(a, dtype=np.int64) for a in (tokens, lengths, drafts, n_accepted, next_tokens)
    ]
    want = advance(*[a.tolist() for a in args], pad_id)
    if literal is not None and literal != want:
        raise RuntimeError("authoring error: a wrong literal in the statement")
    oracle = ctx.expected(ADVANCE, *args, pad_id)
    if [np.asarray(part).tolist() for part in oracle] != list(want):
        raise RuntimeError("authoring error: the reference disagrees with the definition")
    result = ctx.call(ADVANCE, *args, pad_id)
    check.that(
        isinstance(result, (tuple, list)) and len(result) == 3,
        f"{ADVANCE}: expected (new_tokens, new_lengths, cache_lengths), "
        f"got {type(result).__name__}",
    )
    got_tokens = np.asarray(result[0])
    rows = len(want[1])
    got_lengths = int_vector(result[1], rows, f"{ADVANCE}: new_lengths")
    got_cache = int_vector(result[2], rows, f"{ADVANCE}: cache_lengths")
    detail = (
        f"call: {ADVANCE}(tokens, lengths, draft_tokens, n_accepted, next_tokens, pad_id)\n"
        f"  tokens =\n{args[0]}\n  lengths = {args[1].tolist()}\n"
        f"  draft_tokens =\n{args[2]}\n  n_accepted = {args[3].tolist()}\n"
        f"  next_tokens = {args[4].tolist()}\n  pad_id = {pad_id}\n"
        f"expected new_tokens =\n{np.asarray(want[0])}\n"
        f"expected new_lengths = {want[1]}, cache_lengths = {want[2]}\n"
        f"got new_tokens =\n{got_tokens}\ngot new_lengths = {got_lengths}, "
        f"cache_lengths = {got_cache}"
    )
    check.that(got_lengths == want[1], f"{ADVANCE}: wrong new_lengths", detail)
    check.that(got_cache == want[2], f"{ADVANCE}: wrong cache_lengths", detail)
    check.that(
        got_tokens.dtype.kind in "iu" and got_tokens.ndim == 2 and got_tokens.shape[0] == rows,
        f"{ADVANCE}: new_tokens must be an integer matrix with one row per sequence",
        detail,
    )
    check.that(
        got_tokens.shape[1] == max(want[1]),
        f"{ADVANCE}: new_tokens has width {got_tokens.shape[1]}, which is not the longest new row",
        detail,
    )
    for b in range(rows):
        row, n = got_tokens[b].tolist(), want[1][b]
        check.that(
            row[:n] == want[0][b][:n],
            f"{ADVANCE}: row {b} does not hold its old tokens, accepted drafts, and next token",
            detail,
        )
        check.that(
            row[n:] == want[0][b][n:],
            f"{ADVANCE}: row {b} has something other than pad_id after its last token",
            detail,
        )


def check_lossless(ctx, p, q, n_calls, *, bonus_rows=()):
    """Sample every row's drafts from its q, verify in batches, compare each row with p."""
    rows, k, vocab = q.shape
    pe, qe = exact(p), exact(q)
    accepted = [[] for _ in range(rows)]
    first = [[] for _ in range(rows)]
    bonus = [[] for _ in range(rows)]
    q_cdf = np.cumsum(qe, axis=-1)
    for _ in range(n_calls):
        # Inverse CDF per row and position: draft i of row b is a sample from q[b, i].
        drafts = np.minimum((ctx.rng.random((rows, k, 1)) > q_cdf).sum(axis=-1), vocab - 1)
        n, tokens = verify(ctx, p, q, drafts, ctx.rng.random((rows, k)), ctx.rng.random(rows))
        for b in range(rows):
            check.that(
                n[b] == k or tokens[b] != drafts[b, n[b]],
                f"{VERIFY}: after a rejection the next token can never be the rejected draft",
                f"row {b}: draft_tokens = {drafts[b].tolist()}, n_accepted = {n[b]}, "
                f"next token = {tokens[b]}",
            )
            accepted[b].append(n[b])
            first[b].append(int(drafts[b, 0]) if n[b] > 0 else tokens[b])
            if n[b] == k:
                bonus[b].append(tokens[b])
    for b in range(rows):
        beta = np.minimum(pe[b, :k], qe[b]).sum(axis=1)
        reach = np.concatenate([[1.0], np.cumprod(beta)])  # P(n_accepted >= j)
        check.distribution(
            accepted[b],
            reach * np.concatenate([1 - beta, [1.0]]),
            delta=DELTA,
            label=f"n_accepted of row {b}",
        )
        check.distribution(first[b], pe[b, 0], delta=DELTA, label=f"first emitted token of row {b}")
    for b in bonus_rows:
        if len(bonus[b]) < 150:
            raise RuntimeError("authoring error: too few fully accepted rows")
        check.distribution(bonus[b], pe[b, k], delta=DELTA, label=f"bonus token of row {b}")


@case("the worked example: two rows that accept one and two drafts, then the append",
      public=True)  # fmt: skip
def worked_example(ctx):
    check_verify(
        ctx, P, Q, [[1, 1], [0, 2]], [[0.9, 0.7], [0.8, 0.6]], [0.5, 0.45],
        literal=([1, 2], [2, 2]),
    )  # fmt: skip
    check_advance(
        ctx,
        [[11, 12, 13, 77], [21, 22, 23, 24]],
        [3, 4],
        [[1, 1], [0, 2]],
        [1, 2],
        [2, 2],
        9,
        literal=([[11, 12, 13, 1, 2, 9, 9], [21, 22, 23, 24, 0, 2, 2]], [5, 7], [4, 6]),
    )


@case("each row stops at its own first rejection: a grid of uniforms over a 3-row batch",
      public=True)  # fmt: skip
def ragged_grid(ctx):
    # Rejects batch_minimum, counts_all_passes, resamples_from_p, bonus_from_residual,
    # and one_uniform_for_the_batch call by call.
    p, q = models(ctx, 3, 3, 6, closeness=[0.2, 0.6, 1.0])
    checked = 0
    for drafts in ([[0, 1, 2], [3, 4, 5], [1, 1, 4]], [[5, 5, 0], [2, 0, 3], [4, 2, 2]]):
        for u0 in (0.04, 0.5, 0.96):
            for u1 in (0.9, 0.1):
                for ur in ([0.13, 0.71, 0.4], [0.88, 0.07, 0.55]):
                    u_accept = [[u0, u1, 0.3], [u1, u0, 0.6], [0.5, 0.5, 0.5]]
                    args = (p, q, drafts, u_accept, ur)
                    if definition(*args)[2]:
                        check_verify(ctx, *args)
                        checked += 1
    if checked < 16:
        raise RuntimeError("authoring error: too few decisive grid points")


@case("the append is ragged: rows of different lengths accept different numbers of drafts",
      public=True)  # fmt: skip
def ragged_append(ctx):
    # Rejects appends_at_batch_width, keeps_rejected_drafts, copies_stale_padding,
    # lengths_without_next_token, and cache_keeps_next_token. The entries past each
    # row's length are arbitrary, and pad_id 0 is also a real token id.
    tokens = [[5, 6, 7, 8, 9, 3], [5, 6, 41, 42, 43, 44], [5, 6, 7, 8, 51, 52], [5, 0, 0, 0, 0, 0]]
    drafts = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [0, 0, 0, 0]]
    check_advance(ctx, tokens, [6, 2, 4, 1], drafts, [0, 4, 2, 3], [60, 61, 62, 0], 0)
    check_advance(ctx, tokens, [6, 2, 4, 1], drafts, [4, 0, 1, 0], [60, 61, 62, 63], 0)
    check_advance(ctx, tokens, [1, 1, 1, 1], drafts, [0, 0, 0, 0], [60, 61, 62, 63], 99)
    check_advance(ctx, tokens, [6, 6, 6, 6], drafts, [4, 4, 4, 4], [60, 61, 62, 63], 99)


@case("losslessness per row: three rows with different drafts, each follows its own target",
      public=True)  # fmt: skip
def lossless_rows(ctx):
    p, q = models(ctx, 3, 3, 6, closeness=[0.2, 0.6, 0.95])
    check_lossless(ctx, p, q, 400, bonus_rows=(2,))


@case("a batch equals its rows verified one at a time")
def rows_are_independent(ctx):
    p, q = models(ctx, 3, 3, 6, closeness=[0.3, 0.5, 0.9])

    def prop(rng):
        drafts = rng.integers(0, 6, size=(3, 3))
        u_accept = rng.random((3, 3)) * rng.choice([0.3, 1.0], size=(3, 1))
        u_resample = rng.random(3)
        if not definition(p, q, drafts, u_accept, u_resample)[2]:
            return  # too close to a boundary to compare exactly
        got = verify(ctx, p, q, drafts, u_accept, u_resample)
        for b in range(3):
            alone = verify(
                ctx, p[b : b + 1], q[b : b + 1], drafts[b : b + 1], u_accept[b : b + 1],
                u_resample[b : b + 1],
            )  # fmt: skip
            check.that(
                (got[0][b], got[1][b]) == (alone[0][0], alone[1][0]),
                f"{VERIFY}: a row's result changes with the other rows in the batch",
                f"row {b}: in the batch {(got[0][b], got[1][b])}, alone "
                f"{(alone[0][0], alone[1][0])}",
            )

    check.forall(prop, trials=12)


@case("square shapes: a batch of 6 rows, 6 tokens in the vocabulary, K = 5")
def square_shapes(ctx):
    # B = V hides a threshold or a gather that broadcasts along the wrong axis.
    p, q = models(ctx, 6, 5, 6, closeness=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    for _ in range(6):
        check_verify(ctx, p, q, *decisive_batch(ctx, p, q))


@case("rows whose residuals have very different totals are each normalized on their own")
def residual_totals(ctx):
    # Rejects batch_total. Row 0 is rejected with residual total 0.6, row 1 with 0.05.
    p = np.array(
        [
            [[0.1, 0.1, 0.4, 0.4], [0.25, 0.25, 0.25, 0.25]],
            [[0.3, 0.25, 0.25, 0.2], [0.25, 0.25, 0.25, 0.25]],
            [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]],
        ],
        dtype=np.float32,
    )
    q = np.array([[[0.5, 0.3, 0.1, 0.1]], [[0.35, 0.25, 0.2, 0.2]], [[0.4, 0.3, 0.2, 0.1]]],
                 dtype=np.float32)  # fmt: skip
    for ur in ([0.2, 0.2, 0.2], [0.7, 0.7, 0.7], [0.95, 0.3, 0.5], [0.3, 0.95, 0.05]):
        check_verify(ctx, p, q, [[0], [0], [1]], [[0.9], [0.95], [0.5]], ur)


@case("one row, one draft, and a batch in which every row is fully accepted or fully rejected")
def edge_batches(ctx):
    p, q = models(ctx, 1, 1, 6, closeness=0.5)
    for _ in range(4):
        check_verify(ctx, p, q, *decisive_batch(ctx, p, q))
    p, q = models(ctx, 3, 3, 6, closeness=1.0)
    check_verify(ctx, p, p[:, :3].copy(), [[0, 1, 2], [3, 4, 5], [2, 2, 2]],
                 np.full((3, 3), 0.99), [0.1, 0.5, 0.9])  # fmt: skip
    # Drafts the target excludes: rejected at once, even with a uniform of 0.
    p = np.tile(np.array([0.0, 0.5, 0.3, 0.2, 0.0, 0.0], dtype=np.float32), (3, 4, 1))
    q = np.tile(np.array([0.4, 0.1, 0.1, 0.1, 0.2, 0.1], dtype=np.float32), (3, 3, 1))
    check_verify(ctx, p, q, [[0, 1, 1], [4, 1, 1], [5, 1, 1]], np.zeros((3, 3)), [0.1, 0.5, 0.9])
    check_advance(ctx, [[7]], [1], [[3]], [1], [4], 0)
    check_advance(ctx, [[7, 8]], [2], [[3]], [0], [4], 0)


@case("random ragged appends: lengths, accepted counts, and stale entries all vary")
def random_appends(ctx):
    rng = ctx.rng
    for _ in range(40):
        # A short list of shapes: JAX compiles once per shape.
        rows, width, k = SHAPES[int(rng.integers(len(SHAPES)))]
        check_advance(
            ctx,
            rng.integers(0, 64, size=(rows, width)),
            rng.integers(1, width + 1, size=rows),
            rng.integers(0, 64, size=(rows, k)),
            rng.integers(0, k + 1, size=rows),
            rng.integers(0, 64, size=rows),
            int(rng.integers(0, 64)),
        )


@case("several rounds in a row: verify, append, repeat; every row stays its own sequence")
def chained_rounds(ctx):
    p, q = models(ctx, 3, 3, 6, closeness=[0.2, 0.6, 1.0])
    tokens = np.array([[1, 2, 3], [4, 0, 0], [5, 5, 0]], dtype=np.int64)
    lengths = np.array([3, 1, 2], dtype=np.int64)
    history = [[1, 2, 3], [4], [5, 5]]
    for _ in range(4):
        drafts, u_accept, u_resample = decisive_batch(ctx, p, q)
        n, nxt = verify(ctx, p, q, drafts, u_accept, u_resample)
        check_advance(ctx, tokens, lengths, drafts, n, nxt, 0)
        for b in range(3):
            history[b] += [*drafts[b, : n[b]].tolist(), nxt[b]]
        lengths = np.array([len(row) for row in history], dtype=np.int64)
        tokens = np.zeros((3, int(lengths.max())), dtype=np.int64)
        for b in range(3):
            tokens[b, : lengths[b]] = history[b]


@case("a poor draft in every row: the first token of each row still follows its target")
def lossless_poor_drafts(ctx):
    p, q = models(ctx, 3, 3, 6, closeness=[0.0, 0.1, 0.3])
    check_lossless(ctx, p, q, 300)
