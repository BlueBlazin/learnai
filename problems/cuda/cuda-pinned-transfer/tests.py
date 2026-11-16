"""Tests for cuda-pinned-transfer.

Pinning and `non_blocking=True` change no value, and on a CPU they change no timing
either, so the CPU cases judge the *sequence* of operations and the numbers together.

`Prefetcher` gets the kit's `TransferSpy` as `pin` and `to_device`. The spy records
every pin and every transfer, with the label of the batch it belongs to, whether the
transferred tensor came out of `pin`, and whether the call was non-blocking. The
source of batches is a generator that marks the moment each batch is taken, and the
test marks the moment each batch is handed out. From that one ordered log:

- every batch is pinned once and transferred once, from its pinned copy, non-blocking;
- the transfer of batch i + 1 starts before batch i is handed out (the look-ahead);
- batch i + 2 is not taken from the source before batch i is handed out (one ahead,
  not the whole dataset);
- the first transfer starts in the constructor.

On the CPU the spy's "transfer" returns a view of the pinned copy. A prefetcher that
overwrites a pinned buffer whose batch has not been used yet therefore changes that
batch, as a copy still in flight would on a GPU; the values are checked when a batch
is handed out and again after the loop.

`fetch_results` gets a `HostLink`, defined below: a device-to-host copy that, when
non-blocking, returns memory holding NaN (or -1 for integers) until `sync()` runs.

The GPU cases (`requires="cuda"`) were written from the PyTorch 2.15 documentation and
have never run on the machine that validates the catalog.
"""

import numpy as np
import torch

from learnai.judge import case, check
from learnai.kit import TransferSpy

CPU = torch.device("cpu")


# --------------------------------------------------------------------------------------
# Prefetcher
# --------------------------------------------------------------------------------------


def position(order, kind, label):
    """Index of the first (kind, label) event in the log, or None."""
    return order.index((kind, label)) if (kind, label) in order else None


def drive(ctx, arrays, *, lazy=True):
    """Iterate a Prefetcher over `arrays`; check the batches, then the event log."""
    spy = TransferSpy()
    sources = [spy.label(torch.from_numpy(a.copy()), i) for i, a in enumerate(arrays)]
    n = len(sources)
    moved = []

    def to_device(tensor, device, non_blocking=False):
        out = spy.to_device(tensor, device, non_blocking=non_blocking)
        moved.append(out)
        return out

    def generator():
        for i, tensor in enumerate(sources):
            spy.mark(("taken", i))
            yield tensor

    fetcher = ctx.solution.Prefetcher(generator() if lazy else sources, CPU, to_device, spy.pin)
    spy.mark("constructed")
    check.that(
        iter(fetcher) is fetcher,
        "Prefetcher: iter(prefetcher) must return the prefetcher itself",
    )

    got = []
    while True:
        try:
            batch = next(fetcher)
        except StopIteration:
            break
        spy.mark(("handed out", len(got)))
        check.that(
            len(got) < n,
            "Prefetcher: yielded more batches than the source holds",
        )
        check.that(
            isinstance(batch, torch.Tensor),
            f"Prefetcher: yielded a {type(batch).__name__}, not a tensor",
        )
        check.that(
            batch.dtype == sources[len(got)].dtype,
            f"Prefetcher: batch {len(got)} has dtype {batch.dtype}, not the dtype of its source",
        )
        check.match(batch.numpy(), arrays[len(got)], label=f"batch {len(got)}")
        got.append(batch)

    # 1. Every batch, in order, with its values intact after the whole loop.
    check.that(
        len(got) == n,
        f"Prefetcher: yielded {len(got)} batches, which is not the number in the source",
        f"the source holds {n} batches",
    )
    for i, batch in enumerate(got):
        check.match(batch.numpy(), arrays[i], label=f"batch {i} after the loop")
        check.match(sources[i].numpy(), arrays[i], label=f"source batch {i}")
    for _ in range(2):
        check.raises(StopIteration, next, fetcher, label="next() after the last batch")

    # 2. What was handed out is what to_device returned.
    for i, batch in enumerate(got):
        check.that(
            any(batch is out for out in moved),
            f"Prefetcher: batch {i} is not a tensor returned by to_device",
            "hand out the tensor on the device, not the source or its pinned copy",
        )

    # 3. One pin and one transfer per batch, from the pinned copy, non-blocking.
    order = spy.order()
    log = spy.describe()
    for kind, verb in (("pin", "pinned"), ("transfer", "transferred")):
        counts = [order.count((kind, i)) for i in range(n)]
        total = sum(1 for event in spy.events if event.kind == kind)
        check.that(
            counts == [1] * n and total == n,
            f"Prefetcher: {total} tensors were {verb}, and not each batch exactly once",
            f"times each batch was {verb}: {counts}\n\nevent log:\n{log}",
        )
    check.that(
        not spy.unpinned,
        f"Prefetcher: {len(spy.unpinned)} batches were transferred from pageable memory",
        "to_device must receive the tensor that pin() returned; pin() does not work in "
        f"place\n\nevent log:\n{log}",
    )
    check.that(
        not spy.blocking,
        f"Prefetcher: {len(spy.blocking)} transfers were made without non_blocking=True",
        f"event log:\n{log}",
    )

    # 4. The look-ahead: exactly one batch.
    if n:
        check.that(
            position(order, "transfer", 0) < position(order, "mark", "constructed"),
            "Prefetcher: the first transfer had not started when the constructor returned",
            f"event log:\n{log}",
        )
    for i in range(n - 1):
        check.that(
            position(order, "transfer", i + 1) < position(order, "mark", ("handed out", i)),
            f"Prefetcher: batch {i} was handed out before the transfer of batch {i + 1} "
            "had started",
            f"event log:\n{log}",
        )
    for i in range(n - 2):
        handed_out = position(order, "mark", ("handed out", i))
        check.that(
            position(order, "transfer", i + 2) > handed_out,
            f"Prefetcher: the transfer of batch {i + 2} started before batch {i} was "
            "handed out; the look-ahead is one batch",
            f"event log:\n{log}",
        )
        if lazy:
            check.that(
                position(order, "mark", ("taken", i + 2)) > handed_out,
                f"Prefetcher: batch {i + 2} was taken from the source before batch {i} "
                "was handed out; the look-ahead is one batch",
                f"event log:\n{log}",
            )
    return spy


def token_batches(ctx, sizes, width=6):
    return [ctx.randint(0, 1000, (rows, width)) for rows in sizes]


# --------------------------------------------------------------------------------------
# fetch_results
# --------------------------------------------------------------------------------------


class HostLink:
    """A device-to-host link whose non-blocking copies arrive at `sync()`.

    `to_host(tensor, non_blocking=False)` stands for `tensor.to("cpu", non_blocking=...)`
    on a CUDA tensor. A blocking call waits and returns the data. A non-blocking call
    returns at once a tensor of the right shape whose contents are undefined (NaN here,
    -1 for integers) and are filled in, in place, by the next `sync()`.
    """

    def __init__(self):
        self.pending = []
        self.syncs = 0
        self.blocking_copies = 0
        self.copies = 0

    def to_host(self, tensor, non_blocking=False):
        self.copies += 1
        if not non_blocking:
            self.blocking_copies += 1
            return tensor.clone()
        garbage = float("nan") if tensor.is_floating_point() else -1
        dest = torch.full_like(tensor, garbage)
        self.pending.append((dest, tensor.clone()))
        return dest

    def sync(self):
        self.syncs += 1
        for dest, data in self.pending:
            dest.copy_(data)
        self.pending = []

    @property
    def waits(self):
        return self.syncs + self.blocking_copies


def fetch(ctx, arrays):
    link = HostLink()
    outputs = [torch.from_numpy(a.copy()) for a in arrays]
    result = ctx.solution.fetch_results(outputs, link.to_host, link.sync)
    check.that(
        isinstance(result, torch.Tensor),
        f"fetch_results: returned a {type(result).__name__}, not a tensor",
    )
    check.that(
        link.copies >= 1,
        "fetch_results: nothing was copied to the host; the result never went through to_host",
    )
    check.match(result.numpy(), np.concatenate(arrays, axis=0), label="fetch_results")
    check.at_most("host_waits", link.waits, 1, unit="waits")
    check.that(
        not link.pending,
        "fetch_results: returned while non-blocking copies were still in flight",
    )
    return link


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: three batches, the event order of the statement", public=True)
def worked_example(ctx):
    arrays = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]])]
    spy = drive(ctx, arrays)
    want = [
        ("mark", ("taken", 0)),
        ("pin", 0),
        ("transfer", 0),
        ("mark", "constructed"),
        ("mark", ("taken", 1)),
        ("pin", 1),
        ("transfer", 1),
        ("mark", ("handed out", 0)),
        ("mark", ("taken", 2)),
        ("pin", 2),
        ("transfer", 2),
        ("mark", ("handed out", 1)),
        ("mark", ("handed out", 2)),
    ]
    check.that(
        spy.order() == want,
        "Prefetcher: the order of events is not the one the statement gives for this example",
        f"expected:\n{want}\n\ngot:\n{spy.describe()}",
    )


@case("five batches from a generator: pinned, non-blocking, one batch ahead", public=True)
def five_batches(ctx):
    # Rejects on_demand, late_first_transfer, transfers_everything, drops_last,
    # pageable_non_blocking, pin_result_dropped, blocking_transfer, yields_pinned,
    # and reuses_pinned_buffer.
    drive(ctx, token_batches(ctx, [4, 4, 4, 4, 4]))


@case("fetch_results: three outputs, one wait, the data read after it", public=True)
def fetch_three(ctx):
    # Rejects reads_before_sync and sync_before_copies (NaN), blocking_copies and
    # sync_per_output (three waits), and skips_transfer.
    arrays = [ctx.randn(2, 3), ctx.randn(4, 3), ctx.randn(1, 3)]
    fetch(ctx, arrays)


@case("the batches are a list, and the last batch is smaller", public=True)
def list_and_ragged(ctx):
    drive(ctx, token_batches(ctx, [8, 8, 8, 3]), lazy=False)
    drive(ctx, token_batches(ctx, [8, 8, 8, 3]))


@case("an empty source, a single batch, and two batches")
def short_sources(ctx):
    spy = drive(ctx, [])
    check.that(
        not spy.events or spy.order() == [("mark", "constructed")],
        "Prefetcher: pinned or transferred something although the source is empty",
        spy.describe(),
    )
    drive(ctx, [], lazy=False)
    drive(ctx, token_batches(ctx, [5]))
    drive(ctx, token_batches(ctx, [5, 2]))


@case("float batches of several shapes and dtypes keep their values and dtype")
def float_batches(ctx):
    drive(ctx, [ctx.randn(3, 4, 5) for _ in range(4)])
    drive(ctx, [ctx.randn(2, 7).astype(np.float16) for _ in range(3)], lazy=False)
    drive(ctx, [ctx.randn(rows, 2) for rows in (1, 6, 2, 9, 4, 4)])


@case("twelve equal batches: none is changed after it was handed out")
def many_batches(ctx):
    # reuses_pinned_buffer shows here as a batch that turns into its successor.
    drive(ctx, token_batches(ctx, [3] * 12, width=4))


@case("fetch_results: a single output, integer token ids, and many outputs")
def fetch_shapes(ctx):
    fetch(ctx, [ctx.randn(5, 2, 3)])
    fetch(ctx, [ctx.randint(0, 50000, (rows,)) for rows in (3, 1, 4)])
    fetch(ctx, [ctx.randn(2, 8) for _ in range(16)])


@case("random sources: sizes, counts, lists and generators")
def random_sources(ctx):
    for trial in range(6):
        rng = ctx.child_rng(trial)
        sizes = [int(rows) for rows in rng.integers(1, 6, size=int(rng.integers(1, 9)))]
        arrays = [rng.integers(0, 100, size=(rows, 3)) for rows in sizes]
        drive(ctx, arrays, lazy=bool(trial % 2))
        fetch(ctx, [rng.standard_normal((rows, 4)).astype(np.float32) for rows in sizes])


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: batches arrive on the CUDA device from pinned memory", requires="cuda")
def gpu_prefetch(ctx):
    device = torch.device("cuda")
    arrays = token_batches(ctx, [16, 16, 16, 5], width=32)
    seen = []  # (is_pinned, non_blocking) of every transfer

    def to_device(tensor, device, non_blocking=False):
        seen.append((tensor.is_pinned(), bool(non_blocking)))
        return tensor.to(device, non_blocking=non_blocking)

    sources = [torch.from_numpy(a.copy()) for a in arrays]
    fetcher = ctx.solution.Prefetcher(sources, device, to_device, torch.Tensor.pin_memory)
    check.that(len(seen) == 1, "Prefetcher: the constructor must start exactly one transfer")
    got = list(fetcher)
    check.that(
        len(got) == len(arrays),
        f"Prefetcher: yielded {len(got)} batches, which is not the number in the source",
    )
    for i, batch in enumerate(got):
        check.that(batch.is_cuda, f"Prefetcher: batch {i} is on {batch.device}, not on CUDA")
        check.equal(batch.cpu().numpy(), arrays[i], label=f"batch {i}")
    check.that(
        len(seen) == len(arrays) and all(pinned for pinned, _ in seen),
        "Prefetcher: a batch was transferred from memory that is_pinned() reports as pageable",
    )
    check.that(
        all(non_blocking for _, non_blocking in seen),
        "Prefetcher: a transfer was made without non_blocking=True",
    )


@case("on the GPU: results fetched with real non-blocking copies are complete", requires="cuda")
def gpu_fetch(ctx):
    device = torch.device("cuda")
    weight = torch.from_numpy(ctx.randn(256, 256)).to(device)
    inputs = [torch.from_numpy(ctx.randn(rows, 256)).to(device) for rows in (64, 64, 64, 17)]
    outputs = [torch.tanh(x @ weight) for x in inputs]
    want = torch.cat(outputs).cpu().numpy()
    waits = []

    def to_host(tensor, non_blocking=False):
        if not non_blocking:
            waits.append("blocking copy")
        return tensor.to("cpu", non_blocking=non_blocking)

    def sync():
        waits.append("sync")
        torch.cuda.synchronize(device)

    result = ctx.solution.fetch_results(outputs, to_host, sync)
    check.that(isinstance(result, torch.Tensor), "fetch_results: must return a tensor")
    check.that(
        result.device.type == "cpu",
        f"fetch_results: the result is on {result.device}, not on the host",
    )
    check.allclose(result.numpy(), want, rtol=1e-6, atol=1e-7, label="fetch_results")
    check.at_most("host_waits", len(waits), 1, unit="waits")
