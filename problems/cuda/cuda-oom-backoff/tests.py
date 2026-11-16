"""Tests for cuda-oom-backoff.

The model is the kit's `OOMModel`: a callable that raises `torch.OutOfMemoryError`
(the class a real CUDA allocation failure raises) whenever its batch has more than
`max_items` rows, and that logs every call and every `empty_cache()`. Three things make
the GPU behavior observable on a CPU:

- `model.batch_sizes`, the size of every batch tried, is compared with the sequence
  the statement's rules produce (computed here by `plan`, an independent simulation);
- `model.events` shows whether `empty_cache()` ran between a failure and its retry,
  and at no other time;
- `model.counters.live_at_empty_cache` counts `empty_cache()` calls made while the
  failed batch was still referenced. `OOMModel` holds a weak reference to each batch
  that failed; inside an `except` block the exception's traceback keeps the model's
  frame, and so the batch, alive, exactly as it keeps a CUDA tensor alive.

Transient failures (an out-of-memory error on a call that would otherwise fit) and
errors that are not out-of-memory are injected with `errors={call_index: exception}`.
The model refuses to be called more than `CALL_LIMIT` times, so a solution that
retries forever fails quickly instead of running into the time limit.

The outputs are compared with one call of the same function on all items.

The GPU case (`requires="cuda"`) was written from the PyTorch 2.15 documentation and
has never run on the machine that validates the catalog. It runs the same contract
on CUDA tensors with the real `torch.cuda.empty_cache`; it does not provoke a real
out-of-memory error, whose threshold depends on the machine.
"""

import torch

from learnai.judge import case, check
from learnai.kit import OOMModel

FN = "run_batched"
CALL_LIMIT = 80
UNSET = object()


def oom_error():
    return torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (transient)")


class Model(OOMModel):
    """`OOMModel` with a limit on the number of calls."""

    def __call__(self, batch):
        check.that(
            self.counters.calls < CALL_LIMIT,
            f"{FN}: called the model more than {CALL_LIMIT} times for at most 40 items",
            "the same batch is being retried without ever getting smaller, or the loop "
            f"does not advance. batch sizes tried so far: {self.batch_sizes[:12]} ...",
        )
        return super().__call__(batch)


def plan(n, batch_size, fails, grow_after=None):
    """The batch sizes the statement's rules produce, and whether the run ends in an error.

    `fails(call_index, size)` says whether that call runs out of memory.
    """
    sizes, start, size, streak = [], 0, batch_size, 0
    while start < n:
        length = min(size, n - start)
        sizes.append(length)
        if fails(len(sizes) - 1, length):
            if length == 1:
                return sizes, True
            size, streak = length // 2, 0
            continue
        start += length
        if grow_after is not None and size < batch_size:
            streak += 1
            if streak == grow_after:
                size, streak = min(2 * size, batch_size), 0
    return sizes, False


def run(ctx, n, batch_size, max_items, *, grow_after=UNSET, transient=(), device=None, dim=4):
    """Run the solution once and check outputs, batch sizes, and the empty_cache protocol."""
    weight = torch.from_numpy(ctx.randn(dim, 3))
    items = torch.from_numpy(ctx.randn(n, dim))
    if device is not None:
        weight, items = weight.to(device), items.to(device)

    def fn(batch):
        return torch.tanh(batch @ weight)

    model = Model(fn, max_items, errors={index: oom_error() for index in transient})
    cuda_calls = []

    def empty_cache():
        model.empty_cache()
        if device is not None:
            cuda_calls.append(torch.cuda.empty_cache())

    def fails(index, size):
        return index in transient or size > max_items

    want_sizes, raises = plan(n, batch_size, fails, None if grow_after is UNSET else grow_after)
    extra = () if grow_after is UNSET else (grow_after,)
    snapshot = items.clone()
    out = None
    if raises:
        check.raises(
            torch.OutOfMemoryError,
            ctx.solution.run_batched,
            model,
            items,
            batch_size,
            empty_cache,
            *extra,
            label=FN,
        )
    else:
        out = ctx.solution.run_batched(model, items, batch_size, empty_cache, *extra)
        check.that(isinstance(out, torch.Tensor), f"{FN}: returned a {type(out).__name__}")
        if device is not None:
            check.that(out.is_cuda, f"{FN}: the result is on {out.device}, not on the CUDA device")
        # Outputs first: rows missing, duplicated, or out of order.
        check.allclose(
            out.cpu().numpy(), fn(snapshot).cpu().numpy(), rtol=1e-5, atol=1e-6, label=FN
        )
    check.that(bool(torch.equal(items, snapshot)), f"{FN}: modified `items`")

    # The sequence of batches.
    check.equal(model.batch_sizes, want_sizes, label="batch sizes tried")

    # empty_cache: once after every failure that is retried, when the batch is gone.
    events = model.events
    failures = [i for i, event in enumerate(events) if event[0] == "call" and event[2] != "ok"]
    retried = failures[:-1] if raises else failures
    for i in retried:
        check.that(
            i + 1 < len(events) and events[i + 1][0] == "empty_cache",
            f"{FN}: retried after an out-of-memory error without calling empty_cache() first",
            f"event log: {events}",
        )
    check.cost(model.counters, "empty_cache_calls", exactly=len(retried), unit="calls")
    check.that(
        model.counters.live_at_empty_cache == 0,
        f"{FN}: empty_cache() ran {model.counters.live_at_empty_cache} times while the "
        "failed batch was still referenced",
        "Inside the except block the exception holds the traceback, the traceback holds "
        "the frame of the failed call, and the frame holds the batch. A local variable "
        "of your own that names the batch keeps it alive too.",
    )
    return model, out


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: 10 items, batch size 8, the model fits 3", public=True)
def worked_example(ctx):
    # Rejects same_size_forever, restarts_from_first_item, no_empty_cache,
    # empty_cache_in_handler, keeps_batch_alive, forgets_reduced_size,
    # empty_cache_every_batch, and retries_at_the_end.
    model, _ = run(ctx, 10, 8, 3)
    if model.batch_sizes != [8, 4, 2, 2, 2, 2, 2]:
        raise RuntimeError("authoring error: the worked example changed")


@case("the same run with grow_after=2: the size is doubled after two successes", public=True)
def grows_back(ctx):
    # Rejects never_grows.
    model, _ = run(ctx, 10, 8, 3, grow_after=2)
    if model.batch_sizes != [8, 4, 2, 2, 4, 2, 2, 2]:
        raise RuntimeError("authoring error: the worked example changed")


@case("everything fits: one pass, and empty_cache is never called", public=True)
def no_failure(ctx):
    run(ctx, 12, 4, 100)
    run(ctx, 7, 16, 100, grow_after=1)
    run(ctx, 5, 2, 2)


@case("a single item does not fit: the out-of-memory error is raised", public=True)
def single_item_fails(ctx):
    # Rejects halves_to_zero. Sizes tried: 4, 2, 1, then the error; empty_cache twice.
    model, _ = run(ctx, 6, 4, 0)
    if model.batch_sizes != [4, 2, 1]:
        raise RuntimeError("authoring error: the case changed")
    run(ctx, 3, 1, 0)


@case("an error that is not out-of-memory is raised at once, unchanged")
def other_errors_propagate(ctx):
    # Rejects catches_runtime_error: OutOfMemoryError is a subclass of RuntimeError.
    weight = torch.from_numpy(ctx.randn(4, 3))
    items = torch.from_numpy(ctx.randn(12, 4))
    injected = RuntimeError("mat1 and mat2 shapes cannot be multiplied (4x4 and 5x3)")
    model = Model(lambda batch: batch @ weight, 100, errors={1: injected})
    try:
        ctx.solution.run_batched(model, items, 4, model.empty_cache)
    except RuntimeError as exc:
        check.that(
            exc is injected,
            f"{FN}: raised {type(exc).__name__}, which is not the exception the model raised",
        )
    else:
        check.that(False, f"{FN}: swallowed an exception of the model that is not out-of-memory")
    check.equal(model.batch_sizes, [4, 4], label="batch sizes tried")
    check.cost(model.counters, "empty_cache_calls", exactly=0, unit="calls")


@case("a transient failure: the reduced size is kept, or grown back when asked")
def transient_failure(ctx):
    # The second call fails although the model fits 100 items.
    model, _ = run(ctx, 20, 4, 100, transient=(1,))
    if model.batch_sizes != [4, 4, 2, 2, 2, 2, 2, 2, 2, 2]:
        raise RuntimeError("authoring error: the case changed")
    model, _ = run(ctx, 20, 4, 100, transient=(1,), grow_after=2)
    if model.batch_sizes != [4, 4, 2, 2, 4, 4, 4]:
        raise RuntimeError("authoring error: the case changed")
    # A failure in the middle of a streak starts the count again.
    run(ctx, 24, 8, 100, transient=(1, 4), grow_after=3)


@case("growing stops at batch_size, and odd sizes halve downwards")
def growth_limits(ctx):
    # Rejects grows_past_batch_size (5 -> 2 -> 4 -> 5, not 8) and streak_survives_failure.
    model, _ = run(ctx, 30, 5, 100, transient=(0,), grow_after=1)
    if model.batch_sizes[:5] != [5, 2, 4, 5, 5]:
        raise RuntimeError("authoring error: the case changed")
    run(ctx, 23, 7, 3, grow_after=2)
    run(ctx, 16, 8, 100, transient=(0, 2), grow_after=2)
    run(ctx, 31, 6, 2, grow_after=4)


@case("the batch that failed is halved, even when it was a short last slice")
def short_last_slice(ctx):
    # Rejects halves_the_setting: 3 items at size 8 fail, the retry has 1 item, not 3.
    model, _ = run(ctx, 3, 8, 1)
    if model.batch_sizes != [3, 1, 1, 1]:
        raise RuntimeError("authoring error: the case changed")
    run(ctx, 11, 8, 2)
    run(ctx, 9, 8, 100, transient=(1,))


@case("one item, batch size 1, and a batch size larger than the number of items")
def small_inputs(ctx):
    run(ctx, 1, 1, 1)
    run(ctx, 1, 64, 1)
    run(ctx, 2, 64, 1)
    run(ctx, 40, 32, 5, dim=2)


@case("random runs: capacities, transient failures, and growth settings")
def random_runs(ctx):
    for trial in range(12):
        rng = ctx.child_rng(trial)
        n = int(rng.integers(1, 33))
        batch_size = int(rng.integers(1, 17))
        max_items = int(rng.integers(1, 9))
        grow_after = [UNSET, None, 1, 2, 3][int(rng.integers(0, 5))]
        transient = tuple(
            int(i) for i in rng.choice(12, size=int(rng.integers(0, 3)), replace=False)
        )
        run(ctx, n, batch_size, max_items, grow_after=grow_after, transient=transient)


# --------------------------------------------------------------------------------------
# GPU case. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: CUDA items, the real empty_cache, results on the device", requires="cuda")
def gpu_backoff(ctx):
    device = torch.device("cuda")
    run(ctx, 10, 8, 3, device=device)
    run(ctx, 24, 8, 100, transient=(1, 4), grow_after=3, device=device)
    run(ctx, 6, 4, 0, device=device)
