"""Tests for mlx-streaming-generate.

Every case drives the solution's generator to the end and checks, in this order:

1. the stream: the texts, the tokens handed to `detok`, and the log-probabilities,
   against greedy decoding done here eagerly and without a cache (the whole sequence
   through a fresh model at every step, log-softmax in float64 NumPy);
2. where evaluation happened: `EvalSpy` records, in order, every `mx.async_eval`,
   `mx.eval`, and implicit read made by the solution file. The sequence must be
   `async_eval, async_eval, read, async_eval, read, ..., read, read`: step `t + 1` is
   scheduled before token `t` is read, one read per token, no `mx.eval`;
3. the same overlap, without the spy: the `detok` handed to the solution notes how
   many times the model has been called when it is asked for token `t`. With step
   `t + 1` built first, that is `t + 2` (and `n` for the last token);
4. that it streams: when the consumer receives pair `t`, the model has been called
   `min(t + 2, n)` times, and the log-probability it receives has no pending
   operation (`mx.export_to_dot`; an array that was passed to `mx.async_eval` has
   none, from the moment it is scheduled);
5. the work asked of the model, as in `cached-greedy-decode`.
"""

import io

import mlx.core as mx
import numpy as np

from whetstone.judge import case, check
from whetstone.kit import EvalSpy, MLXTinyLM

FN = "stream_generate"

# The worked example of the statement.
PROMPT = [0, 17, 42, 5]
CONTINUATION = [28, 34, 31, 15, 1, 48, 46, 5]
LOGPROBS = [-0.1615, -1.4226, -1.3849, -1.9084, -1.9787, -1.8911, -2.1124, -1.5798]


def pending_ops(*arrays):
    """Number of operations not yet scheduled or computed behind `arrays`."""
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return dot.getvalue().count("shape=rectangle")


class Detok:
    """The detokenizer handed to the solution: one piece of text per token id.

    It records every call and how many times the model had been called by then.
    """

    def __init__(self, lm):
        self.lm = lm
        self.tokens = []
        self.forward_calls = []

    @staticmethod
    def piece(token):
        return f"[{token}]"

    def __call__(self, token):
        if type(token) is not int:
            raise TypeError(f"detok takes a Python int, got {type(token).__name__}")
        self.tokens.append(token)
        self.forward_calls.append(self.lm.counters.forward_calls)
        return self.piece(token)


def eager_greedy(seed, prompt, n, **options):
    """The definition: uncached greedy decoding, evaluated at once, scored in float64."""
    lm = MLXTinyLM(seed, **options)
    sequence, tokens, logprobs = list(prompt), [], []
    for _ in range(n):
        logits, _ = lm(mx.array(sequence))
        row = np.array(logits[-1]).astype(np.float64)
        top = np.sort(row)[-2:]
        if top[1] - top[0] < 1e-3:
            raise RuntimeError("authoring error: the two largest logits are too close")
        scores = row - row.max() - np.log(np.exp(row - row.max()).sum())
        token = int(row.argmax())
        tokens.append(token)
        logprobs.append(float(scores[token]))
        sequence.append(token)
    return tokens, logprobs


def expected_kinds(n):
    """async_eval of step 0, then for each token: async_eval of the next step, read."""
    kinds = ["async_eval"]
    for index in range(n):
        if index + 1 < n:
            kinds.append("async_eval")
        kinds.append("read")
    return kinds


def first_order_violation(kinds, n):
    """Index of the first token that was read before the next step was scheduled."""
    scheduled = reads = 0
    for kind in kinds:
        if kind == "async_eval":
            scheduled += 1
        elif kind == "read":
            # Token `reads` needs step reads + 1 scheduled, i.e. reads + 2 async_evals
            # (the last token has no step after it).
            if scheduled < min(reads + 2, n):
                return reads
            reads += 1
    return None


def run(ctx, seed, prompt, n, expected=None, **options):
    """Drive the solution's stream on a fresh model and check stream, order, and cost."""
    want_tokens, want_logprobs = eager_greedy(seed, prompt, n, **options)
    if expected is not None and want_tokens != expected:
        raise RuntimeError("authoring error: the worked example changed")
    lm = MLXTinyLM(seed, **options)
    detok = Detok(lm)
    received = []  # (item, forward calls when it arrived, pending operations behind it)
    with EvalSpy() as spy:
        stream = ctx.solution.stream_generate(lm, list(prompt), n, detok)
        check.that(
            hasattr(stream, "__next__"),
            f"{FN}: must be a generator (an iterator that yields one pair per token)",
            f"got {type(stream).__name__}",
        )
        for item in stream:
            arrays = [x for x in item if isinstance(x, mx.array)] if isinstance(item, tuple) else []
            pending = pending_ops(*arrays) if arrays else 0
            received.append((item, lm.counters.forward_calls, pending))
            if len(received) > n + 4:
                break

    # 1. The stream.
    check.that(
        all(
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], mx.array)
            and item[1].ndim == 0
            and item[1].dtype == mx.float32
            for item, _, _ in received
        ),
        f"{FN}: must yield pairs (text, logprob) of a str and a scalar float32 mx.array",
        f"first item: {received[0][0]!r}" if received else "nothing was yielded",
    )
    check.equal(
        [item[0] for item, _, _ in received], [Detok.piece(t) for t in want_tokens], label="texts"
    )
    check.equal(detok.tokens, want_tokens, label="tokens passed to detok")
    check.allclose(
        np.array([item[1].item() for item, _, _ in received], dtype=np.float32),
        np.array(want_logprobs, dtype=np.float32),
        atol=1e-3,
        label="logprobs",
    )

    # 2. Where evaluation happened, and in which order.
    kinds = spy.kinds()
    check.that(
        spy.evals == 0,
        f"{FN}: called mx.eval {spy.evals} time(s); it blocks until the step is computed. "
        "Schedule with mx.async_eval",
        spy.describe(),
    )
    late = first_order_violation(kinds, n)
    check.that(
        late is None,
        f"{FN}: token {late} was read before the step after it was scheduled with "
        "mx.async_eval, so the GPU is idle while Python handles the token",
        spy.describe(),
    )
    check.at_most("async_evals", spy.async_evals, n)
    check.at_most("reads", spy.implicit, n)
    check.that(
        kinds == expected_kinds(n),
        f"{FN}: the sequence of evaluations ({spy.async_evals} async_eval, {spy.implicit} "
        "reads) is not one async_eval per step and one read per token, each step "
        "scheduled before the token before it is read",
        f"expected {expected_kinds(n)}\ngot      {kinds}\n{spy.describe()}",
    )

    # 3. The overlap as detok saw it: step t + 1 was built before token t was detokenized.
    built = [min(index + 2, n) for index in range(n)]
    behind = [i for i, (got, want) in enumerate(zip(detok.forward_calls, built)) if got < want]
    check.that(
        not behind,
        f"{FN}: when detok was called for token {behind[0] if behind else 0}, the model "
        "had not yet been called for the step after it",
        f"model calls at each detok call: {detok.forward_calls}\nexpected {built}",
    )

    # 4. It streams, and what it yields is scheduled.
    arrived = [calls for _, calls, _ in received]
    check.that(
        arrived == built,
        f"{FN}: pairs do not arrive as they are produced: when pair "
        f"{next((i for i, (a, b) in enumerate(zip(arrived, built)) if a != b), 0)} "
        "arrived, the model had been called a number of times that is not one step ahead",
        f"model calls when each pair arrived: {arrived}\nexpected {built}",
    )
    lazy = [index for index, (_, _, pending) in enumerate(received) if pending]
    check.that(
        not lazy,
        f"{FN}: the log-probability of token {lazy[0] if lazy else 0} was yielded with "
        "operations still pending: it was not part of any mx.async_eval",
        f"pending operations behind each yielded logprob: {[p for _, _, p in received]}",
    )

    # 5. The work asked of the model.
    check.cost(lm.counters, "forward_calls", at_most=n)
    check.cost(lm.counters, "tokens_processed", at_most=len(prompt) + n - 1, unit="tokens")
    return received


@case("the worked example: 8 tokens streamed from a 4-token prompt", public=True)
def worked_example(ctx):
    received = run(ctx, 3, PROMPT, 8, expected=CONTINUATION)
    got = [item[1].item() for item, _, _ in received]
    if np.abs(np.array(got) - np.array(LOGPROBS)).max() > 1e-3:
        raise RuntimeError("authoring error: the worked example's log-probabilities changed")


@case("n = 1: one async_eval, one read, and no step after the last token", public=True)
def single_token(ctx):
    # Rejects extra_speculative_step in its smallest form.
    run(ctx, 3, PROMPT, 1)
    run(ctx, 9, [0, 12, 40], 1)


@case("n = 2: the second step is scheduled before the first token is read", public=True)
def two_tokens(ctx):
    run(ctx, 2, [0, 50, 51], 2)


@case("24 tokens: every pair arrives while the next step is already running", public=True)
def twenty_four_tokens(ctx):
    run(ctx, 4, [0, 8, 8, 8], 24)


# Hidden cases. The mutants differ in the rule they break, not in the input they need:
# each fails the public cases too, for the reason its BUG line gives.


@case("another model: 60 tokens from a 6-token prompt")
def long_generation(ctx):
    run(ctx, 5, [0, 33, 2, 60, 14, 7], 60)


@case("a 40-token prompt and 3 new tokens: the prompt goes in as one call")
def long_prompt(ctx):
    run(ctx, 7, [0, *range(20, 59)], 3)


@case("a single-token prompt")
def single_token_prompt(ctx):
    run(ctx, 11, [0], 20)


@case("a model with grouped-query attention and sinusoidal positions")
def other_architecture(ctx):
    run(ctx, 6, [0, 3, 9, 27], 16, n_kv_heads=2, rope=False)


@case("a consumer that stops early: closing the stream after 3 of 10 pairs")
def early_stop(ctx):
    want_tokens, _ = eager_greedy(8, [0, 21, 4], 10)
    lm = MLXTinyLM(8)
    detok = Detok(lm)
    stream = ctx.solution.stream_generate(lm, [0, 21, 4], 10, detok)
    texts = []
    for text, _ in stream:
        texts.append(text)
        if len(texts) == 3:
            break
    stream.close()
    check.equal(texts, [Detok.piece(t) for t in want_tokens[:3]], label="texts")
    # Three tokens delivered, the fourth step in flight: nothing beyond that was built.
    check.cost(lm.counters, "forward_calls", at_most=4)
