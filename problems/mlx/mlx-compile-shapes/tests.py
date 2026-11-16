"""Tests for mlx-compile-shapes.

Every case calls the solution's step on a sequence of prompts, evaluating the token
and `state["counts"]` after each call as a caller would, and checks, in this order:

1. the result: every token, exactly, and the counts after every call, against the
   definition computed here without `mx.compile` and without padding (each prompt, as
   it is, through a twin model; the penalty and the argmax in numpy);
2. tracing: the model handed to the solution records the length of the token array
   every time its Python body runs. Under `mx.compile` the body runs only when the
   function is traced, so this is the list of traces, an exact count: at most one per
   bucket in use, and each on exactly the bucket's number of tokens;
3. evaluation: `EvalSpy` watches the solution file during every call of `step`.
"""

import mlx.core as mx
import numpy as np

from learnai.judge import case, check
from learnai.kit import EvalSpy, MLXTinyLM

FN = "step"
VOCAB = 64
BUCKETS = (16, 32, 64)


def bucket_of(length):
    return next(size for size in BUCKETS if size >= length)


class RecordingLM(MLXTinyLM):
    """An MLXTinyLM that notes how many tokens each run of its Python body was given."""

    def __init__(self, seed, **options):
        super().__init__(seed, **options)
        self.traced_lengths = []

    def __call__(self, tokens, cache=None, **options):
        self.traced_lengths.append(int(tokens.shape[-1]))
        return super().__call__(tokens, cache, **options)


def definition(seed, prompts, penalty, *, counts=None, resets=None, **options):
    """Tokens and the counts after each call: unpadded, uncompiled, in numpy."""
    lm = MLXTinyLM(seed, **options)
    counts = np.zeros(VOCAB, np.float32) if counts is None else counts.copy()
    tokens, history = [], []
    for index, prompt in enumerate(prompts):
        if resets and index in resets:
            counts = resets[index].copy()
        logits, _ = lm(mx.array(prompt))
        row = np.array(logits[-1]) - np.float32(penalty) * counts
        top = np.sort(row)[-2:]
        if top[1] - top[0] < 1e-3:
            raise RuntimeError("authoring error: the two largest adjusted logits are too close")
        token = int(row.argmax())
        counts[token] += 1
        tokens.append(token)
        history.append(counts.copy())
    return tokens, history


def drive(ctx, seed, prompts, penalty, *, counts=None, resets=None, lm=None, **options):
    """Run the solution on `prompts`; check result, traces, and evaluation."""
    want_tokens, want_counts = definition(
        seed, prompts, penalty, counts=counts, resets=resets, **options
    )
    lm = RecordingLM(seed, **options) if lm is None else lm
    already = len(lm.traced_lengths)
    start = mx.zeros(VOCAB) if counts is None else mx.array(counts)
    mx.eval(start)
    state = {"counts": start}
    step = ctx.solution.make_sampler_step(lm, state, penalty)
    check.that(callable(step), "make_sampler_step: must return a function")
    check.that(
        len(lm.traced_lengths) == already,
        "make_sampler_step: ran the model before step was called",
        "mx.compile traces on the first call; make_sampler_step only builds the function",
    )

    got_tokens, got_counts, problems = [], [], []
    with EvalSpy() as spy:
        for index, prompt in enumerate(prompts):
            if resets and index in resets:
                state["counts"] = mx.array(resets[index])  # the caller replaces the state
            token = step(mx.array(prompt, dtype=mx.int32))
            if not isinstance(token, mx.array):
                problems.append(f"call {index + 1}: returned a {type(token).__name__}")
                continue
            if not isinstance(state.get("counts"), mx.array):
                problems.append(f"call {index + 1}: state['counts'] is no longer an mx.array")
                continue
            try:
                mx.eval(token, state["counts"])  # the caller's evaluation
            except RuntimeError as error:
                # MLX's own message is several lines long; say here what it means.
                check.that(
                    False,
                    f"{FN}: the token or state['counts'] cannot be evaluated after a call "
                    "(an array created while tracing escaped from the compiled function)",
                    f"call {index + 1}: {error}",
                )
            if token.shape != () or token.dtype != mx.int32:
                problems.append(f"call {index + 1}: token has shape {token.shape}, {token.dtype}")
                continue
            got_tokens.append(token.item())
            got_counts.append(np.array(state["counts"]))

    # 1. The result.
    check.that(
        not problems,
        f"{FN}: must return an int32 scalar mx.array and keep state['counts'] an mx.array "
        f"({len(problems)} of {len(prompts)} calls did not)",
        "\n".join(problems),
    )
    check.equal(got_tokens, want_tokens, label="tokens")
    for index, (got, want) in enumerate(zip(got_counts, want_counts, strict=True)):
        check.that(
            got.shape == (VOCAB,) and got.dtype == np.float32,
            f"{FN}: state['counts'] has shape {got.shape} and dtype {got.dtype} after a call",
            f"expected a float32 array of shape ({VOCAB},), after call {index + 1}",
        )
        check.allclose(got, want, atol=1e-6, label=f"state['counts'] after call {index + 1}")

    # 2. Tracing: one run of the model's body per bucket in use, on the bucket's length.
    traced = lm.traced_lengths[already:]
    needed = sorted({bucket_of(len(prompt)) for prompt in prompts})
    ctx.metric("traces", len(traced), limit=len(needed))
    for length in traced:
        check.that(
            length in needed,
            f"{FN}: the model was traced on {length} tokens, which is not the bucket of any "
            "prompt in this run",
            f"traced on {traced}; the prompts need the buckets {needed}",
        )
    check.that(
        len(traced) <= len(needed),
        f"{FN}: the model's Python body ran {len(traced)} times, more than once per bucket",
        f"traced on {traced}; the prompts need the buckets {needed}",
    )

    # 3. Evaluation.
    seen = spy.evals + spy.async_evals + spy.implicit
    check.that(
        seen == 0,
        f"{FN}: evaluated {seen} time(s) (mx.eval, .item(), int(), ...); the step must only "
        "build a graph",
        spy.describe(),
    )
    return lm, state


def prompts_of(ctx, lengths):
    return [ctx.randint(0, VOCAB, (length,)).tolist() for length in lengths]


@case("the worked example: one prompt three times and a longer one, penalty 1.5", public=True)
def worked_example(ctx):
    short, long = [0, 17, 42, 5], list(range(20, 40))
    prompts = [short, short, long, short]
    tokens, _ = definition(1, prompts, 1.5)
    if tokens != [55, 55, 12, 20]:
        raise RuntimeError("authoring error: the worked example changed")
    lm, _ = drive(ctx, 1, prompts, 1.5)
    if lm.traced_lengths != [16, 32]:
        raise RuntimeError("authoring error: the reference's traces changed")


@case("40 prompts of 12 distinct lengths: at most 3 traces", public=True)
def forty_prompts(ctx):
    # Rejects retrace_per_length and python_int_position (12 traces) and shapeless.
    distinct = [1, 3, 7, 12, 16, 17, 20, 25, 31, 33, 40, 64]
    lengths = distinct + [int(n) for n in ctx.rng.choice(distinct, size=28)]
    lengths = [lengths[i] for i in ctx.rng.permutation(len(lengths))]
    drive(ctx, 2, prompts_of(ctx, lengths), 0.5)


@case("lengths at the bucket edges: 16, 17, 32, 33, 64, and 1", public=True)
def bucket_edges(ctx):
    # A prompt that fills its bucket has no padding, so reads_last_position and
    # pads_left are right there and wrong on every other length.
    drive(ctx, 3, prompts_of(ctx, [16, 17, 32, 33, 64, 1, 16, 32]), 0.5)


@case("the caller replaces state['counts'] between calls", public=True)
def caller_replaces_state(ctx):
    # Rejects state_frozen: the compiled graph must read the dict at every call.
    prompts = prompts_of(ctx, [5, 5, 9, 9, 9, 14])
    tokens, _ = definition(4, prompts, 2.0)
    banned = np.zeros(VOCAB, np.float32)
    banned[tokens[3]] = 50.0  # make the token that call 4 would pick impossible
    resets = {3: banned, 5: np.zeros(VOCAB, np.float32)}
    replaced, _ = definition(4, prompts, 2.0, resets=resets)
    if replaced[3] == tokens[3]:
        raise RuntimeError("authoring error: replacing the counts should change call 4")
    drive(ctx, 4, prompts, 2.0, resets=resets)


# Hidden cases.


@case("short prompts only: one trace, on 16 tokens")
def short_prompts_only(ctx):
    # Rejects one_bucket: padding everything to 64 quadruples the work of every call.
    drive(ctx, 5, prompts_of(ctx, [2, 9, 4, 16, 11, 1, 7, 13, 5, 3]), 1.0)


@case("the same prompt ten times: the penalty changes the token as the counts grow")
def repeated_prompt(ctx):
    prompt = prompts_of(ctx, [6])[0]
    tokens, _ = definition(6, [prompt] * 10, 1.0)
    if len(set(tokens)) < 3:
        raise RuntimeError("authoring error: the penalty should move the argmax")
    drive(ctx, 6, [prompt] * 10, 1.0)


@case("prompts made of token 0, and prompts that end in token 0")
def zeros_in_prompts(ctx):
    # Rejects length_from_padding: token 0 is an ordinary token, not a marker.
    prompts = [
        [0],
        [0, 0, 0],
        [7, 0],
        [0] * 16,
        [9, 30, 0, 0],
        [0] * 20,
        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [41, 0, 12, 0],
    ]
    drive(ctx, 7, prompts, 0.5)


@case("counts that do not start at zero, and a penalty of 0")
def other_starts(ctx):
    start = ctx.uniform(VOCAB, low=0.0, high=3.0).round().astype(np.float32)
    drive(ctx, 8, prompts_of(ctx, [3, 18, 3, 40, 18, 8]), 1.0, counts=start)
    drive(ctx, 8, prompts_of(ctx, [3, 18, 3, 40, 18, 8]), 0.0)


@case("two samplers on one model keep separate states and separate traces")
def two_samplers(ctx):
    lm, first = drive(ctx, 9, prompts_of(ctx, [4, 22, 10]), 1.0)
    _, second = drive(ctx, 9, prompts_of(ctx, [6, 30, 12]), 1.0, lm=lm)
    check.that(
        first is not second and float(np.array(first["counts"]).sum()) == 3.0,
        "make_sampler_step: a second sampler changed the state of the first",
        "each call of make_sampler_step gets its own state dict",
    )


@case("grouped-query attention with sinusoidal positions, 20 prompts")
def other_architecture(ctx):
    lengths = [int(n) for n in ctx.rng.integers(1, 65, size=20)]
    drive(ctx, 10, prompts_of(ctx, lengths), 0.5, n_kv_heads=2, rope=False)
