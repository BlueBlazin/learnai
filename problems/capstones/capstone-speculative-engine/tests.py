"""Tests for capstone-speculative-engine.

Every expectation comes from `simulate`, which runs the statement's round procedure
in numpy with no cache at all (every model call gets the whole sequence) and in
float64. It returns the tokens and the rounds, and it refuses fixtures on which a
decision is close: an argmax whose two largest logits are within 1e-3, an acceptance
test or an inverse-CDF threshold within a relative 1e-3 or 1e-4 of its boundary. So
the float32 arithmetic of a solution, and the way it splits its model calls, cannot
change a result. For greedy runs the tokens are also compared with the target's own
greedy decode.

Costs are the kit counters of the two models, with limits derived from the simulated
rounds (R rounds, D drafted tokens, D_m of them by the draft model):

    target forward_calls     <= R
    target tokens_processed  <= P + D + R
    draft tokens_processed   <= P + N + D_m

Statistical cases use a pair of very small models (8 tokens) so that the exact
distribution of the first emitted tokens can be enumerated. There are two of them
with four checks each (three marginals and a joint), within the budget of ten checks
at the default delta of 1e-10.
"""

import numpy as np

from learnai.judge import case, check
from learnai.kit import draft_target_pair

S1 = "1: greedy speculation"
S2 = "2: exact sampling"
S3 = "3: adaptive draft length"
S4 = "4: prompt lookup"
S5 = "5: batches"

MARGIN = 1e-3
TINY = {"vocab_size": 8, "d_model": 16, "n_layers": 1, "n_heads": 2}

# The worked example of stage 1 (the pair and prompt of `speculative-decode-loop`).
PROMPT = [0, 17, 42, 5]
CONTINUATION = [42, 62, 33, 33, 40, 32, 45, 37, 40, 33, 40, 55]


class NearTie(Exception):
    pass


class Plain:
    """A do-nothing adapter: with it, the kit models speak numpy."""

    name = "numpy"

    def is_tensor(self, value):
        return False

    def tensor(self, value, dtype=None):
        return value

    def numpy(self, value):
        return value


_pairs = {}


def pair(seed, agreement, fw=None, **size):
    """(draft, target) with fresh counters, for `fw` or (by default) for numpy."""
    key = (seed, agreement, tuple(sorted(size.items())))
    if key not in _pairs:
        _pairs[key] = draft_target_pair(Plain(), seed=seed, agreement=agreement, **size)
    draft, target = _pairs[key]
    fw = Plain() if fw is None else fw
    return draft.with_framework(fw), target.with_framework(fw)


# -- the specification, in numpy ----------------------------------------------------------


def lookup_draft(context, max_ngram, min_ngram, k):
    length = len(context)
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = context[length - n :]
        for start in range(length - n):
            if context[start : start + n] == suffix:
                return context[start + n : start + n + k]
    return []


def softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def top(row):
    order = np.sort(row)
    if order[-1] - order[-2] < MARGIN:
        raise NearTie("argmax")
    return int(row.argmax())


def pick(weights, u):
    cdf = np.cumsum(weights)
    if np.abs(cdf - u * cdf[-1]).min() < 1e-4 * cdf[-1]:
        raise NearTie("inverse CDF")
    return int((cdf <= u * cdf[-1]).sum())


def simulate(draft, target, prompt, max_new_tokens, *, k=4, adaptive=False, lookup=None,
             uniforms=None):  # fmt: skip
    """The statement's rounds without any cache. Returns (tokens, rounds)."""
    vocab = target.vocab_size
    seq, rounds = list(prompt), []
    while len(seq) - len(prompt) < max_new_tokens:
        limit = min(k, max_new_tokens - (len(seq) - len(prompt)) - 1)
        u = None if uniforms is None else uniforms[len(rounds)]
        proposer, drafts, q = "draft", [], []
        found = lookup_draft(seq, *lookup, limit) if limit and lookup else []
        if found:
            proposer, drafts, q = "lookup", list(found), [np.eye(vocab)[x] for x in found]
        while proposer == "draft" and len(drafts) < limit:
            row = draft(seq + drafts)[0][-1]
            q.append(softmax(row))
            drafts.append(top(row) if u is None else pick(q[-1], float(u[len(drafts), 0])))
        logits = target(seq + drafts)[0][len(seq) - 1 :]
        n = 0
        if u is None:
            while n < len(drafts) and drafts[n] == top(logits[n]):
                n += 1
            token = top(logits[n])
        else:
            p = softmax(logits)
            while n < len(drafts):
                x = drafts[n]
                left, right = float(u[n, 1]) * q[n][x], p[n][x]
                if abs(left - right) < MARGIN * max(left, right):
                    raise NearTie("acceptance test")
                if not left < right:
                    break
                n += 1
            weights = np.maximum(p[n] - q[n], 0.0) if n < len(drafts) else p[n]
            token = pick(weights, float(u[n, 2]))
        seq += [*drafts[:n], token]
        rounds.append((proposer, len(drafts), n))
        if adaptive and drafts:
            k = k + 2 if n == len(drafts) else max(1, k - 1)
    return seq[len(prompt) :], rounds


def target_greedy(target, prompt, max_new_tokens):
    """The definition of a greedy result: the target's own decode, without any draft."""
    seq = list(prompt)
    for _ in range(max_new_tokens):
        seq.append(int(target(seq)[0][-1].argmax()))
    return seq[len(prompt) :]


def safe_uniforms(rng, model, prompt, max_new_tokens, **options):
    """Uniforms (N, N, 3) on which no decision of the run is close to its boundary."""
    for _ in range(200):
        uniforms = rng.random((max_new_tokens, max_new_tokens, 3)).astype(np.float32)
        try:
            simulate(*pair(*model[:2], **model[2]), prompt, max_new_tokens, uniforms=uniforms,
                     **options)  # fmt: skip
        except NearTie:
            continue
        return uniforms
    raise RuntimeError("authoring error: no safe uniforms found")


# -- running the solution -----------------------------------------------------------------


def as_rounds(value):
    try:
        return [(r[0], int(r[1]), int(r[2])) for r in value]
    except (TypeError, IndexError, ValueError):
        return value


def describe(rounds):
    return "\n".join(
        f"  round {i + 1}: {proposer}, {drafted} drafted, {accepted} accepted"
        for i, (proposer, drafted, accepted) in enumerate(rounds)
    )


def check_tokens(got, want, label="generated tokens"):
    check.that(
        isinstance(got, list) and all(type(token) is int for token in got),
        f"{label}: must be a list of Python ints",
        f"got {got!r}",
    )
    check.equal(got, want, label=label)


def run(ctx, model, prompt, max_new_tokens, uniforms=None, expected=None, engine=None, **options):
    """One `generate` call: tokens, rounds, then the three counters. Returns the rounds."""
    seed, agreement, size = model
    try:
        want, rounds = simulate(*pair(seed, agreement, **size), prompt, max_new_tokens,
                                uniforms=uniforms, **options)  # fmt: skip
    except NearTie as error:
        raise RuntimeError(f"authoring error: a near-tie ({error}) in a fixture") from None
    if uniforms is None and want != target_greedy(pair(seed, agreement, **size)[1], prompt,
                                                  max_new_tokens):  # fmt: skip
        raise RuntimeError("authoring error: the rounds do not reproduce the greedy decode")
    if expected is not None and want != expected:
        raise RuntimeError("authoring error: the statement's example changed")
    draft, target = pair(seed, agreement, ctx.fw, **size)
    if engine is None:
        engine = ctx.solution.SpeculativeEngine(target, draft, **options)
    else:
        engine = engine(target, draft)
    if uniforms is None:
        got = engine.generate(list(prompt), max_new_tokens)
    else:
        got = engine.generate(list(prompt), max_new_tokens, ctx.fw.tensor(uniforms))
    check_tokens(got, want)
    got_rounds = as_rounds(getattr(engine, "rounds", None))
    check.that(
        got_rounds == rounds,
        "engine.rounds is not the list of rounds the statement's procedure takes",
        f"expected {len(rounds)} rounds:\n{describe(rounds)}\ngot: {got_rounds!r}",
    )
    n_rounds = len(rounds)
    drafted = sum(r[1] for r in rounds)
    by_model = sum(r[1] for r in rounds if r[0] == "draft")
    counters = target.counters
    check.at_most("target_forward_calls", counters.forward_calls, n_rounds, unit="calls")
    check.at_most("target_tokens_processed", counters.tokens_processed,
                  len(prompt) + drafted + n_rounds, unit="tokens")  # fmt: skip
    check.at_most("draft_tokens_processed", draft.counters.tokens_processed,
                  len(prompt) + max_new_tokens + by_model, unit="tokens")  # fmt: skip
    return rounds


BIG = {}  # the default TinyLM: 64 tokens


# -- stage 1: greedy speculation with rollback --------------------------------------------


@case("the worked example: 12 tokens in 4 rounds with k = 4", public=True, stage=S1)
def worked_example(ctx):
    rounds = run(ctx, (3, 0.9, BIG), PROMPT, 12, k=4, expected=CONTINUATION)
    if rounds != [("draft", 4, 0), ("draft", 4, 4), ("draft", 4, 4), ("draft", 0, 0)]:
        raise RuntimeError("authoring error: the rounds of the statement changed")


@case("a draft that mostly agrees: 40 tokens in fewer than 20 target calls", public=True,
      stage=S1)  # fmt: skip
def high_agreement(ctx):
    # Rejects refeeds_whole_sequence: the token limits are linear in the output.
    for seed, prompt, k in [(3, [0, 8, 8, 8], 4), (4, [0, 12, 40], 6)]:
        rounds = run(ctx, (seed, 0.9, BIG), prompt, 40, k=k)
        if len(rounds) >= 20:
            raise RuntimeError("authoring error: this fixture needs too many rounds")


@case("a draft that is often wrong: every rejection rolls both caches back", public=True,
      stage=S1)  # fmt: skip
def low_agreement(ctx):
    # Rejects stale_target_cache (wrong tokens) and stale_draft_cache (right tokens,
    # but the draft continues from rejected tokens, so the rounds are not the specified
    # ones and there are more of them).
    for seed, prompt, k in [(5, [0, 30, 31, 32], 4), (5, [0, 9, 9], 3)]:
        rounds = run(ctx, (seed, 0.5, BIG), prompt, 32, k=k)
        if sum(1 for r in rounds if r[2] < r[1]) < 8:
            raise RuntimeError("authoring error: this fixture has too few rejections")


@case("draft lengths 1 and 8, and a single new token", stage=S1)
def edge_lengths(ctx):
    run(ctx, (4, 0.9, BIG), [0, 21, 22, 23, 24], 30, k=1)
    run(ctx, (3, 0.9, BIG), [0, 3, 4], 30, k=8)
    run(ctx, (4, 0.9, BIG), [5], 1, k=4)
    run(ctx, (4, 0.9, BIG), [5], 2, k=4)


@case("one engine, several calls: nothing carries over", stage=S1)
def engine_reuse(ctx):
    draft, target = pair(3, 0.9, ctx.fw)
    engine = ctx.solution.SpeculativeEngine(target, draft, k=3)
    numpy_draft, numpy_target = pair(3, 0.9)
    for prompt, n in ([0, 3, 4], 20), ([0, 8, 8, 8], 12), ([0, 3, 4], 20):
        want, rounds = simulate(numpy_draft, numpy_target, prompt, n, k=3)
        check_tokens(engine.generate(list(prompt), n), want)
        check.that(
            as_rounds(engine.rounds) == rounds,
            "engine.rounds does not describe the last call only",
            f"expected:\n{describe(rounds)}\ngot: {as_rounds(engine.rounds)!r}",
        )


# -- stage 2: sampling ----------------------------------------------------------------------


def exact_sampling(ctx, model, prompt, max_new_tokens, n_runs, **options):
    for _ in range(n_runs):
        uniforms = safe_uniforms(ctx.rng, model, prompt, max_new_tokens, **options)
        run(ctx, model, prompt, max_new_tokens, uniforms=uniforms, **options)


def emitted_distribution(target, prompt, n_positions):
    """Exact marginals of the first tokens the target samples, and the joint of the first two."""
    vocab = target.vocab_size
    prefixes, weights, marginals, joint = [[]], np.ones(1), [], None
    for position in range(n_positions):
        tokens = np.array([list(prompt) + prefix for prefix in prefixes])
        probs = softmax(target(tokens)[0][:, -1]) * weights[:, None]
        marginals.append(probs.sum(axis=0))
        if position == 1:
            joint = probs.reshape(-1)
        prefixes = [prefix + [v] for prefix in prefixes for v in range(vocab)]
        weights = probs.reshape(-1)
    return marginals, joint


def statistical(ctx, seed, agreement, prompt, n_calls, n_positions=3, **options):
    draft, target = pair(seed, agreement, ctx.fw, **TINY)
    marginals, joint = emitted_distribution(pair(seed, agreement, **TINY)[1], prompt, n_positions)
    engine = ctx.solution.SpeculativeEngine(target, draft, **options)
    uniforms = ctx.rng.random((n_calls, n_positions, n_positions, 3)).astype(np.float32)
    samples = []
    for i in range(n_calls):
        tokens = engine.generate(list(prompt), n_positions, ctx.fw.tensor(uniforms[i]))
        check.that(
            isinstance(tokens, list)
            and len(tokens) == n_positions
            and all(type(t) is int and 0 <= t < target.vocab_size for t in tokens),
            "generate: did not return max_new_tokens token ids of the vocabulary",
            f"got {tokens!r}",
        )
        samples.append(tokens)
    samples = np.array(samples)
    for position in range(n_positions):
        check.distribution(samples[:, position].tolist(), marginals[position],
                           label=f"token {position + 1} of the output")  # fmt: skip
    pairs = (samples[:, 0] * target.vocab_size + samples[:, 1]).tolist()
    check.distribution(pairs, joint, label="first two tokens of the output")


@case("supplied uniforms decide every draft, acceptance, and resample", public=True, stage=S2)
def sampling_exact(ctx):
    exact_sampling(ctx, (3, 0.9, BIG), PROMPT, 12, 3, k=4)
    exact_sampling(ctx, (5, 0.5, BIG), [0, 30, 31, 32], 12, 3, k=3)


@case("the emitted tokens are distributed as the target's own samples", public=True, stage=S2)
def sampling_distribution(ctx):
    # Rejects residual_unnormalized and draft_argmax_when_sampling.
    statistical(ctx, 2, 0.6, [0, 3, 5], 600, k=2)


@case("sampling with other draft lengths, to the last token", stage=S2)
def sampling_exact_more(ctx):
    exact_sampling(ctx, (4, 0.9, BIG), [0, 21, 22, 23, 24], 10, 3, k=1)
    exact_sampling(ctx, (5, 0.5, BIG), [0, 9, 9], 10, 3, k=8)
    exact_sampling(ctx, (2, 0.6, TINY), [0, 3, 5], 8, 4, k=2)
    exact_sampling(ctx, (4, 0.9, BIG), [5], 1, 2, k=4)


# -- stage 3: adaptive draft length -------------------------------------------------------


@case("a draft that agrees: k grows, and 48 tokens take 7 target calls", public=True, stage=S3)
def adaptive_grows(ctx):
    # Rejects fixed_k: with k = 4 throughout, the same output takes 10 target calls.
    rounds = run(ctx, (3, 0.95, BIG), [0, 10, 23, 33], 48, k=4, adaptive=True)
    fixed = simulate(*pair(3, 0.95), [0, 10, 23, 33], 48, k=4)[1]
    if (len(rounds), len(fixed)) != (7, 10):
        raise RuntimeError(f"authoring error: {len(rounds)} adaptive and {len(fixed)} fixed rounds")


@case("a draft that is often wrong: k shrinks, and the draft model does less work", public=True,
      stage=S3)  # fmt: skip
def adaptive_shrinks(ctx):
    rounds = run(ctx, (4, 0.5, BIG), [0, 11, 24, 33], 48, k=4, adaptive=True)
    fixed = simulate(*pair(4, 0.5), [0, 11, 24, 33], 48, k=4)[1]
    drafted, fixed_drafted = sum(r[1] for r in rounds), sum(r[1] for r in fixed)
    if drafted > 0.7 * fixed_drafted:
        raise RuntimeError(f"authoring error: {drafted} adaptive and {fixed_drafted} fixed drafts")


@case("other starting lengths and models", stage=S3)
def adaptive_more(ctx):
    run(ctx, (3, 0.95, BIG), [0, 14, 27, 33], 48, k=1, adaptive=True)
    run(ctx, (4, 0.5, BIG), [0, 15, 28, 33], 48, k=6, adaptive=True)
    run(ctx, (5, 0.5, BIG), [0, 8, 21, 33], 20, k=2, adaptive=True)


@case("adaptive length while sampling", stage=S3)
def adaptive_sampling(ctx):
    exact_sampling(ctx, (3, 0.95, BIG), [0, 10, 23, 33], 14, 3, k=2, adaptive=True)
    exact_sampling(ctx, (4, 0.5, BIG), [0, 11, 24, 33], 14, 3, k=4, adaptive=True)


# -- stage 4: prompt lookup as a second proposer ------------------------------------------


@case("a repetitive output: lookup drafts 11 of 14 rounds without the draft model", public=True,
      stage=S4)  # fmt: skip
def lookup_repetitive(ctx):
    # Rejects lookup_ignored (the draft model does every round's work) and
    # lookup_latest_occurrence.
    rounds = run(ctx, (36, 0.6, BIG), [0, 5, 5, 9], 60, k=4, lookup=(3, 2))
    if sum(1 for r in rounds if r[0] == "lookup") != 11 or len(rounds) != 14:
        raise RuntimeError("authoring error: the rounds of the statement changed")


@case("copied tokens can be wrong, and a round without a match uses the draft model",
      public=True, stage=S4)  # fmt: skip
def lookup_falls_back(ctx):
    # Rejects lookup_unverified: here the target rejects some of the copied tokens.
    rounds = run(ctx, (29, 0.6, BIG), [0, 11, 12, 13], 60, k=4, lookup=(3, 1))
    proposers = {r[0] for r in rounds}
    if proposers != {"draft", "lookup"}:
        raise RuntimeError("authoring error: this fixture must use both proposers")


@case("other n-gram ranges, with and without adaptive length", stage=S4)
def lookup_more(ctx):
    run(ctx, (36, 0.6, BIG), [0, 60, 4, 10], 48, k=4, lookup=(3, 2))
    run(ctx, (29, 0.6, BIG), [0, 46, 20, 35], 48, k=3, lookup=(2, 1), adaptive=True)
    run(ctx, (29, 0.6, BIG), [0, 11, 21, 62], 48, k=6, lookup=(4, 1))
    run(ctx, (3, 0.9, BIG), PROMPT, 12, k=4, lookup=(8, 5))  # never matches


@case("lookup drafts while sampling: exact runs", stage=S4)
def lookup_sampling_exact(ctx):
    exact_sampling(ctx, (2, 0.6, TINY), [0, 3, 5, 3], 10, 4, k=3, lookup=(2, 1))
    exact_sampling(ctx, (36, 0.6, BIG), [0, 5, 5, 9, 5, 5], 12, 3, k=4, lookup=(3, 1))


@case("lookup drafts while sampling: the distribution is still the target's", stage=S4)
def lookup_sampling_distribution(ctx):
    # Rejects lookup_unverified and lookup_residual_keeps_token.
    statistical(ctx, 2, 0.6, [0, 3, 5, 3], 600, k=2, lookup=(2, 1))


# -- stage 5: a batch with ragged acceptance ----------------------------------------------


def batch_target_tokens(prompt_len, max_new_tokens, per_row):
    """Tokens the target processes when rows share one cache cut to the shortest valid row."""
    rows = range(len(per_row))
    length = [prompt_len] * len(per_row)
    done = [0] * len(per_row)
    active, cached, total = list(rows), 0, 0
    while active:
        total += len(active) * max(length[b] - cached + per_row[b][done[b]][1] for b in active)
        valid = {}
        for b in active:
            _, _, accepted = per_row[b][done[b]]
            valid[b] = length[b] + accepted
            length[b] += accepted + 1
            done[b] += 1
        active = [b for b in active if length[b] - prompt_len < max_new_tokens]
        if active:
            cached = min(valid[b] for b in active)
    return total


def run_batch(ctx, seed, agreement, prompts, max_new_tokens, k):
    numpy_draft, numpy_target = pair(seed, agreement)
    want, per_row = [], []
    for prompt in prompts:
        try:
            tokens, rounds = simulate(numpy_draft, numpy_target, prompt, max_new_tokens, k=k)
        except NearTie as error:
            raise RuntimeError(f"authoring error: a near-tie ({error}) in a fixture") from None
        if tokens != target_greedy(numpy_target, prompt, max_new_tokens):
            raise RuntimeError("authoring error: the rounds do not reproduce the greedy decode")
        want.append(tokens)
        per_row.append(rounds)
    draft, target = pair(seed, agreement, ctx.fw)
    engine = ctx.solution.SpeculativeEngine(target, draft, k=k)
    got = engine.generate_batch([list(prompt) for prompt in prompts], max_new_tokens)
    check.that(
        isinstance(got, list) and len(got) == len(prompts),
        "generate_batch: must return one list of tokens per prompt",
        f"got {got!r}",
    )
    for b, (row, wanted) in enumerate(zip(got, want, strict=True)):
        check_tokens(row, wanted, label=f"tokens of row {b}")
    most = max(len(rounds) for rounds in per_row)
    prompt_len = len(prompts[0])
    check.at_most("target_forward_calls", target.counters.forward_calls, most, unit="calls")
    limit = int(1.25 * batch_target_tokens(prompt_len, max_new_tokens, per_row))
    check.at_most("target_tokens_processed", target.counters.tokens_processed, limit, unit="tokens")
    draft_limit = sum(prompt_len + max_new_tokens + sum(r[1] for r in rounds) for rounds in per_row)
    check.at_most("draft_tokens_processed", draft.counters.tokens_processed, draft_limit,
                  unit="tokens")  # fmt: skip
    return per_row


BATCH_PROMPTS = [[0, 23, 13, 42, 38, 9], [0, 57, 42, 51, 25, 13]]


@case("two rows, 40 tokens each: no more target calls than the slower row needs alone",
      public=True, stage=S5)  # fmt: skip
def batch_two_rows(ctx):
    # Rejects rows_cut_to_batch_minimum (right tokens in 31 target calls instead of 21),
    # batch_one_row_at_a_time (21 + 19 calls), and verifies_from_padded_end.
    per_row = run_batch(ctx, 4, 0.5, BATCH_PROMPTS, 40, 4)
    ragged = sum(1 for a, b in zip(*per_row, strict=False) if a[2] != b[2])
    if ragged < 8 or [len(rounds) for rounds in per_row] != [21, 19]:
        raise RuntimeError("authoring error: the rounds of the statement changed")


@case("rows finish in different rounds; one row; three rows", stage=S5)
def batch_more(ctx):
    prompts = [[0, 33, 41, 49, 59, 27], [0, 29, 31, 61, 14, 32], [0, 62, 59, 43, 62, 56]]
    per_row = run_batch(ctx, 5, 0.5, prompts, 24, 4)
    if len({len(rounds) for rounds in per_row}) < 3:
        raise RuntimeError("authoring error: the rows of this fixture finish together")
    run_batch(ctx, 4, 0.9, [[0, 12, 40]], 16, 6)
    run_batch(ctx, 5, 0.5, [[0, 30, 31, 32], [0, 8, 21, 33]], 16, 3)
    run_batch(ctx, 4, 0.9, [[5], [6]], 1, 4)
