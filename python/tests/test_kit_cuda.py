"""The CUDA-on-CPU kit: SyncSpy, DeviceSpy, FakeDevice, OOMModel, FakeCudaGraph,
CompileCounter, TransferSpy. Everything here runs on CPU tensors."""

import gc
import importlib.util
import textwrap

import numpy as np
import pytest
import torch

from learnai.kit import (
    CompileCounter,
    DeviceSpy,
    FakeCudaGraph,
    FakeDevice,
    OOMModel,
    RealDevice,
    SyncSpy,
    TransferSpy,
)

SOLUTION = textwrap.dedent(
    """
    import numpy as np
    import torch


    def sync_free(logits, n, check_every):
        out = torch.empty(n, dtype=torch.long)
        done = torch.zeros((), dtype=torch.bool)
        for i in range(n):
            token = torch.argmax(logits[i])
            out[i] = token
            done = done | (token == 0)
            if (i + 1) % check_every == 0 and done.item():
                break
        return out.cpu()


    def item_per_token(logits, n):
        out = []
        for i in range(n):
            out.append(torch.argmax(logits[i]).item())
        return out


    def python_eos_check(logits, n):
        out = torch.empty(n, dtype=torch.long)
        for i in range(n):
            token = torch.argmax(logits[i])
            out[i] = token
            if token == 0:
                break
        return out


    def every_kind(x):
        a = x.sum().item()
        b = x.tolist()
        c = bool(x[0] > 0)
        d = int(x[0])
        e = float(x[0])
        print(x[0])
        f = f"{x[0]:.2f}"
        g = x[0] in x
        h = list(range(torch.tensor(3)))
        i = x.equal(x)
        j = torch.allclose(x, x)
        k = not x[0]
        l = x.cpu()
        m = x.numpy()
        n = np.asarray(x)
        o = x.nonzero()
        p = x[x > 0]
        q = torch.where(x > 0)
        r = torch.masked_select(x, x > 0)
        while x[0] < 0:
            pass
        return a


    def quiet(x):
        y = (x * 2).to(x.device)
        z = torch.where(x > 0, x, 0.0)
        return y[1:] + z[:-1], len(x), x.shape[0], x[torch.tensor([0, 1])]


    HOST = torch.device("cpu")


    def slices(row, n):
        a = row[:n]
        b = row[None, 1:n]
        row[:n] = 0.0
        return a, b, row[1:3], row[n]


    def names_the_cpu(x):
        a = x.to("cpu")
        b = x.to(device="cpu")
        c = x.to(torch.device("cpu"))
        d = x.to(HOST)
        e = x.float().to(
            device="cpu:0",
            non_blocking=True,
        )
        return a, b, c, d, e


    def moves_without_naming_the_cpu(x, device, other):
        where = "cpu"
        if x.ndim > 5:
            where = other.device
        return (x.to(device), x.to(other.device), x.to(other), x.to(torch.float16),
                x.to(device=device, dtype=torch.float32), x.to(where))


    def to_device(x, device):
        return x.to(device)


    def makes_tensors(x, device):
        a = torch.arange(x.shape[0])
        b = torch.arange(x.shape[0], device=device)
        c = torch.zeros(3, device=None)
        d = torch.ones_like(x)
        e = x.new_zeros(2)
        f = torch.tensor([1.0], device=x.device)
        return a, b, c, d, e, f


    def time_region(fn, dev, n):
        fn()
        dev.sync()
        times = []
        for _ in range(n):
            start = dev.clock()
            fn()
            dev.sync()
            times.append(dev.clock() - start)
        return sorted(times)[len(times) // 2]


    def time_region_no_sync(fn, dev, n):
        fn()
        times = []
        for _ in range(n):
            start = dev.clock()
            fn()
            times.append(dev.clock() - start)
        return sorted(times)[len(times) // 2]


    def run_batched(model, items, batch_size, empty_cache):
        out, start = [], 0
        while start < len(items):
            batch = items[start : start + batch_size]
            failed = False
            try:
                out.append(model(batch))
            except torch.OutOfMemoryError:
                failed = True
            if failed:
                del batch
                empty_cache()
                if batch_size == 1:
                    raise RuntimeError("a single item does not fit")
                batch_size = max(1, batch_size // 2)
                continue
            start += len(batch)
        return torch.cat(out)


    def run_batched_in_except(model, items, batch_size, empty_cache):
        out, start = [], 0
        while start < len(items):
            batch = items[start : start + batch_size]
            try:
                out.append(model(batch))
                start += len(batch)
            except RuntimeError:
                empty_cache()
                batch_size = max(1, batch_size // 2)
        return torch.cat(out)
    """
)


@pytest.fixture
def solution(tmp_path):
    path = tmp_path / "torch_solution.py"
    path.write_text(SOLUTION)
    spec = importlib.util.spec_from_file_location("cuda_kit_solution", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logits(n=12, eos_at=None):
    rows = np.random.default_rng(0).standard_normal((n, 8)).astype(np.float32)
    rows[:, 0] = -10.0
    if eos_at is not None:
        rows[eos_at, 0] = 10.0
    return torch.tensor(rows)


# -- SyncSpy ------------------------------------------------------------------------------


def test_syncspy_counts_one_read_per_check_and_the_final_transfer(solution):
    with SyncSpy(file=solution.__file__) as spy:
        out = solution.sync_free(logits(), 12, 4)
    assert out.shape == (12,)
    assert [(e.kind, e.what) for e in spy.events] == [("read", "item")] * 3 + [("transfer", "cpu")]
    assert (spy.syncs, spy.reads, spy.transfers, spy.shape_syncs) == (4, 3, 1, 0)
    assert "read (item)" in spy.describe() and "transfer (cpu)" in spy.describe()


def test_syncspy_catches_the_per_token_item_and_the_python_eos_check(solution):
    with SyncSpy(file=solution.__file__) as spy:
        solution.item_per_token(logits(), 12)
    assert spy.syncs == 12 and {e.what for e in spy.events} == {"item"}
    # `if token == eos:` makes no visible call; the spy still sees it, once per step.
    with SyncSpy(file=solution.__file__) as spy:
        solution.python_eos_check(logits(), 12)
    assert spy.syncs == 12 and {e.what for e in spy.events} == {"bool"}
    assert len({e.line for e in spy.events}) == 1


def test_syncspy_sees_every_kind_of_host_synchronization_once(solution, capsys):
    x = torch.tensor([1.5, -2.0, 3.0])
    with SyncSpy(file=solution.__file__) as spy:
        solution.every_kind(x)
    capsys.readouterr()
    assert [e.what for e in spy.events] == [
        "item", "tolist", "bool", "int", "float", "print", "format", "in", "index", "equal",
        "allclose", "bool", "cpu", "numpy", "np.asarray", "nonzero", "boolean-mask index",
        "where(condition)", "masked_select", "bool",
    ]  # fmt: skip
    assert (spy.reads, spy.transfers, spy.shape_syncs) == (13, 3, 4)


def test_syncspy_ignores_device_side_work_and_other_files(solution):
    x = torch.tensor([1.5, -2.0, 3.0])
    with SyncSpy(file=solution.__file__) as spy:
        solution.quiet(x)
        # Reads made here, by the test, are not the solution's.
        assert x.sum().item() == 2.5 and bool(x[0] > 0) and x.tolist()
        print(x)
    assert spy.syncs == 0


def test_syncspy_sees_a_tensor_used_as_a_slice_bound(solution):
    row, n = torch.arange(10.0), torch.tensor(3)
    with SyncSpy(file=solution.__file__) as spy:
        a, b, _, _ = solution.slices(row, n)
    assert a.shape == (3,) and b.shape == (1, 2)
    # Two slice reads, one slice write (seen through __index__); a slice with plain
    # ints and a 0-d tensor used as an index are not host reads the spy can prove.
    assert [(e.kind, e.what) for e in spy.events] == [
        ("read", "slice bound"), ("read", "slice bound"), ("read", "index"),
    ]  # fmt: skip
    assert len({e.line for e in spy.events}) == 3


def test_syncspy_counts_to_cpu_only_where_the_source_names_the_cpu(solution):
    x = torch.tensor([1.5, -2.0, 3.0])
    with SyncSpy(file=solution.__file__) as spy:
        solution.names_the_cpu(x)
    assert [(e.kind, e.what) for e in spy.events] == [("transfer", "to(cpu)")] * 5
    assert len({e.line for e in spy.events}) == 5
    # The device a test passes in is the CPU too, as an object or as a string: moving
    # to it, to another tensor's device, or to a dtype is not a host transfer.
    for device in (torch.device("cpu"), "cpu"):
        with SyncSpy(file=solution.__file__) as spy:
            solution.moves_without_naming_the_cpu(x, device, torch.zeros(2))
        assert spy.syncs == 0, spy.describe()
    # The test's own .to("cpu") is not the solution's.
    with SyncSpy(file=solution.__file__) as spy:
        x.to("cpu")
    assert spy.syncs == 0


def test_syncspy_counts_to_cpu_from_a_tensor_that_is_really_elsewhere(solution):
    meta = torch.empty(3, device="meta")
    with SyncSpy(file=solution.__file__) as spy:
        solution.to_device(meta, torch.device("meta"))
        solution.to_device(meta, meta)
        solution.to_device(meta, torch.float16)
    assert spy.syncs == 0
    # From a real non-CPU tensor the target decides, however it is spelled. (The meta
    # device has no data, so the copy itself then fails, as it should.)
    for target in (torch.device("cpu"), "cpu", torch.zeros(1)):
        with SyncSpy(file=solution.__file__) as spy, pytest.raises(NotImplementedError):
            solution.to_device(meta, target)
        assert [(e.kind, e.what) for e in spy.events] == [("transfer", "to(cpu)")]


def test_syncspy_restores_torch_exactly(solution):
    before = {name: torch.Tensor.__dict__.get(name) for name in ("__bool__", "item", "__repr__")}
    functions = (torch.where, torch.nonzero, torch.equal)
    with pytest.raises(ZeroDivisionError), SyncSpy(file=solution.__file__):
        1 / 0  # noqa: B018
    after = {name: torch.Tensor.__dict__.get(name) for name in before}
    assert after == before
    assert torch.Tensor.to is torch._C.TensorBase.to
    assert (torch.where, torch.nonzero, torch.equal) == functions
    assert torch.tensor(3).item() == 3 and bool(torch.tensor(True))


def test_syncspy_needs_a_file_outside_a_case():
    with pytest.raises(ValueError, match="needs file="):
        SyncSpy()


# -- DeviceSpy ----------------------------------------------------------------------------


def test_devicespy_finds_factories_without_a_device(solution):
    x = torch.ones(4)
    arange = torch.arange
    with DeviceSpy(file=solution.__file__) as spy:
        solution.makes_tensors(x, torch.device("cpu"))
        torch.arange(3)  # the test's own call is not watched
    assert [(e.what, e.explicit) for e in spy.events] == [
        ("arange", False), ("arange", True), ("zeros", False), ("tensor", True),
    ]  # fmt: skip
    assert [e.what for e in spy.implicit] == ["arange", "zeros"]
    assert "WITHOUT device=" in spy.describe()
    assert torch.arange is arange


def test_the_meta_device_is_a_second_device_on_any_machine():
    # Documented in PROBLEM_AUTHORING ("CUDA problems"): mixing devices fails on the CPU
    # too when one of them is the meta device.
    x = torch.randn(2, 5).to("meta")
    with pytest.raises(RuntimeError, match="not on the expected device"):
        x + torch.arange(5)
    assert (x + torch.arange(5, device=x.device)).device.type == "meta"


# -- FakeDevice ---------------------------------------------------------------------------


def test_fakedevice_times_only_what_was_waited_for(solution):
    dev = FakeDevice()
    fn = dev.kernel(0.004, first_call_s=0.050)
    assert solution.time_region(fn, dev, 5) == pytest.approx(0.004, rel=1e-9)
    assert dev.counters.syncs == 6 and dev.counters.launches == 6 and dev.pending_s == 0

    # No sync before stopping the timer: the work is queued, nothing is measured.
    dev = FakeDevice()
    assert solution.time_region_no_sync(dev.kernel(0.004, first_call_s=0.050), dev, 5) == 0.0
    assert dev.pending_s == pytest.approx(0.050 + 5 * 0.004)


def test_fakedevice_bills_stale_work_and_warmup_to_whoever_syncs_first():
    dev = FakeDevice()
    dev.launch(0.1)  # queued by "someone else" before the region
    start = dev.clock()
    dev.launch(0.004)
    dev.sync()
    assert dev.clock() - start == pytest.approx(0.104, rel=1e-9)

    dev = FakeDevice()
    fn = dev.kernel(0.004, first_call_s=0.050)
    start = dev.clock()
    fn()
    dev.sync()
    assert dev.clock() - start == pytest.approx(0.050, rel=1e-9)  # warm-up included


def test_fakedevice_runs_concurrently_with_the_host():
    dev = FakeDevice(launch_s=0.001)
    dev.launch(0.010)  # device busy until 0.010; the host moved on to 0.001
    assert dev.clock() == pytest.approx(0.001)
    dev.launch(0.010)  # starts when the device is free (0.010), ends at 0.020
    assert dev.pending_s == pytest.approx(0.018)
    dev.sync()
    assert dev.clock() == pytest.approx(0.020) and dev.pending_s == 0
    assert [kind for kind, _ in dev.events] == ["launch", "clock", "launch", "sync", "clock"]
    with pytest.raises(ValueError):
        dev.launch(-1)


def test_realdevice_on_the_cpu_is_a_plain_timer():
    dev = RealDevice("cpu")
    dev.sync()
    assert dev.clock() <= dev.clock()


# -- OOMModel -----------------------------------------------------------------------------


def test_oommodel_raises_torchs_out_of_memory_error_above_the_limit():
    model = OOMModel(lambda x: x * 2, max_items=3)
    assert model(torch.ones(3)).tolist() == [2.0, 2.0, 2.0]
    with pytest.raises(torch.OutOfMemoryError, match="CUDA out of memory"):
        model(torch.ones(4))
    assert issubclass(torch.OutOfMemoryError, RuntimeError)
    assert torch.cuda.OutOfMemoryError is torch.OutOfMemoryError
    assert model.batch_sizes == [3, 4]
    assert model.counters.snapshot() == {
        "calls": 2, "ooms": 1, "items": 3, "empty_cache_calls": 0, "live_at_empty_cache": 0,
    }  # fmt: skip


def test_oommodel_records_the_backoff_sequence_and_empty_cache(solution):
    model = OOMModel(lambda x: x * 2, max_items=3)
    items = torch.arange(10, dtype=torch.float32)
    out = solution.run_batched(model, items, 8, model.empty_cache)
    assert out.tolist() == (items * 2).tolist()
    assert model.batch_sizes == [8, 4, 2, 2, 2, 2, 2]
    assert model.events[:5] == [
        ("call", 8, "oom"), ("empty_cache", 0), ("call", 4, "oom"), ("empty_cache", 0),
        ("call", 2, "ok"),
    ]  # fmt: skip
    assert model.counters.live_at_empty_cache == 0 and model.counters.items == 10


def test_oommodel_notices_empty_cache_while_the_failed_batch_is_alive(solution):
    # empty_cache() inside the `except` block: the exception still holds the batch.
    model = OOMModel(lambda x: x * 2, max_items=3)
    items = torch.arange(10, dtype=torch.float32)
    solution.run_batched_in_except(model, items, 8, model.empty_cache)
    assert model.counters.empty_cache_calls == 2
    assert model.counters.live_at_empty_cache == 2
    assert ("empty_cache", 1) in model.events


def test_oommodel_can_raise_another_error_and_a_single_item_can_fail(solution):
    boom = RuntimeError("shape mismatch")
    model = OOMModel(lambda x: x, max_items=4, errors={1: boom})
    with pytest.raises(RuntimeError, match="shape mismatch"):
        solution.run_batched(model, torch.ones(8), 4, model.empty_cache)
    assert model.events == [("call", 4, "ok"), ("call", 4, "error")]

    model = OOMModel(lambda x: x, max_items=0)
    with pytest.raises(RuntimeError, match="single item"):
        solution.run_batched(model, torch.ones(4), 2, model.empty_cache)
    assert model.batch_sizes == [2, 1]
    # A list cannot be weakly referenced: it is simply not tracked.
    with pytest.raises(torch.OutOfMemoryError):
        model([1, 2])
    gc.collect()
    model.empty_cache()
    assert model.counters.live_at_empty_cache == 0


# -- FakeCudaGraph ------------------------------------------------------------------------


def step_fn(x, w):
    return torch.relu(x @ w) + 1.0


def test_fake_graph_replays_on_the_captured_memory():
    w = torch.randn(4, 4)
    static_in = torch.randn(2, 4)
    graph = FakeCudaGraph()
    assert not graph.capturing and not graph.captured
    graph.capture_begin()
    assert graph.capturing
    static_out = step_fn(static_in, w)
    graph.capture_end()
    assert graph.captured and not graph.capturing and graph.n_ops >= 3

    new = torch.randn(2, 4)
    static_in.copy_(new)
    out_ptr = static_out.data_ptr()
    graph.replay()
    assert torch.equal(static_out, step_fn(new, w))
    assert static_out.data_ptr() == out_ptr  # the output is overwritten in place
    assert graph.counters.snapshot() == {"captures": 1, "replays": 1}


def test_fake_graph_rebinding_gives_stale_output_and_missing_clone_aliases():
    w = torch.randn(4, 4)
    static_in = torch.randn(2, 4)
    first = static_in.clone()
    graph = FakeCudaGraph()
    graph.capture_begin()
    static_out = step_fn(static_in, w)
    graph.capture_end()

    # Rebinding the name instead of copying into the placeholder: replay cannot see it.
    static_in = torch.randn(2, 4)
    graph.replay()
    assert torch.equal(static_out, step_fn(first, w))

    # Returning the static output without clone(): the next replay changes it.
    kept = static_out
    cloned = static_out.clone()
    captured_input = graph._ops[0][1][0]
    captured_input.copy_(torch.randn(2, 4))
    graph.replay()
    assert not torch.equal(kept, cloned)


def test_fake_graph_replays_in_place_updates_and_views():
    buffer = torch.zeros(4, 3)
    row = torch.zeros(3)
    pos = torch.tensor([0])
    graph = FakeCudaGraph()
    graph.capture_begin()
    buffer.index_copy_(0, pos, row[None, :] * 2)
    total = buffer[:2].sum(0)
    graph.capture_end()
    for step in range(3):
        row.copy_(torch.full((3,), float(step + 1)))
        pos.fill_(step)
        graph.replay()
        assert buffer[step].tolist() == [2.0 * (step + 1)] * 3
    assert total.tolist() == [6.0] * 3  # rows 0 and 1: 2 + 4


def test_fake_graph_refuses_host_reads_during_capture_and_misuse():
    graph = FakeCudaGraph()
    with pytest.raises(RuntimeError, match="replay\\(\\) called before capture_end"):
        graph.replay()
    with pytest.raises(RuntimeError, match="before capture_begin"):
        graph.capture_end()
    x = torch.ones(3)
    graph.capture_begin()
    with pytest.raises(RuntimeError, match="already capturing"):
        graph.capture_begin()
    with pytest.raises(RuntimeError, match="not permitted when stream is capturing"):
        (x * 2).sum().item()
    graph.capture_end()
    with pytest.raises(RuntimeError, match="already owns a captured graph"):
        graph.capture_begin()
    graph.reset()
    assert not graph.captured and graph.n_ops == 0
    # The dispatch mode is gone: ordinary code is unaffected.
    assert (x * 2).sum().item() == 6.0


@pytest.mark.parametrize("mode", [torch.inference_mode, torch.no_grad, None])
def test_fake_graph_replays_whatever_grad_mode_the_capture_ran_in(mode):
    layer = torch.nn.Linear(4, 4)
    x = torch.randn(2, 4)
    graph = FakeCudaGraph()

    def capture():
        graph.capture_begin()
        out = torch.relu(layer(x))
        graph.capture_end()
        return out

    if mode is None:
        y = capture()
    else:
        with mode():
            y = capture()
    new = torch.randn(2, 4)
    x.copy_(new)
    graph.replay()  # outside the mode the capture ran in
    assert torch.allclose(y, torch.relu(layer(new)))


def test_fake_graph_rejects_a_shape_that_depends_on_the_data():
    x = torch.tensor([1.0, -1.0, 2.0, -2.0])
    graph = FakeCudaGraph()
    graph.capture_begin()
    positive = x[x > 0]
    graph.capture_end()
    assert positive.shape == (2,)
    x.copy_(torch.tensor([1.0, 1.0, 2.0, -2.0]))
    with pytest.raises(RuntimeError, match="shapes cannot depend on the data"):
        graph.replay()


def test_fake_graph_works_with_torch_cuda_graph_while_patched():
    real = torch.cuda.graph
    graph = FakeCudaGraph()
    x = torch.ones(3)
    with graph.patched():
        with torch.cuda.graph(graph):
            assert graph.capturing
            y = x * 3
        assert graph.captured
    assert torch.cuda.graph is real
    x.fill_(2.0)
    graph.replay()
    assert y.tolist() == [6.0, 6.0, 6.0]


def test_torch_cuda_graph_surface_matches_what_the_fake_imitates():
    # The names the fake stands in for exist in the pinned torch (it cannot be
    # constructed without CUDA, so only the surface is checked).
    for name in ("capture_begin", "capture_end", "replay", "reset"):
        assert callable(getattr(torch.cuda.CUDAGraph, name))
        assert callable(getattr(FakeCudaGraph, name))
    import inspect

    parameters = inspect.signature(torch.cuda.CUDAGraph.capture_begin).parameters
    assert {"pool", "capture_error_mode"} <= set(parameters)


# -- CompileCounter -----------------------------------------------------------------------


def test_compile_counter_counts_compilations_and_graph_breaks():
    def step(x, pos):
        return x * 2 + pos

    def branchy(x):
        if x.sum() > 0:
            return x + 1
        return x - 1

    with CompileCounter() as cc:
        compiled = torch.compile(step, backend="eager")
        for pos in range(6):
            compiled(torch.ones(3), torch.tensor(pos))
        assert cc.snapshot().compilations == 1
    assert (cc.compilations, cc.graphs, cc.graph_breaks) == (1, 1, 0)

    # A Python int that changes: specialized on the first value, recompiled once with
    # the value made symbolic (torch 2.15's automatic dynamic shapes).
    with CompileCounter() as cc:
        compiled = torch.compile(step, backend="eager")
        for pos in range(6):
            compiled(torch.ones(3), pos)
    assert cc.compilations == 2

    with CompileCounter() as cc:
        compiled = torch.compile(branchy, backend="eager")
        compiled(torch.ones(3))
    assert cc.graph_breaks == 1 and cc.compilations == 2
    assert "Data-dependent branching" in cc.break_reasons[0]
    assert "graph breaks: 1" in cc.describe()

    # Each block starts from zero.
    with CompileCounter() as cc:
        pass
    assert (cc.compilations, cc.graphs, cc.graph_breaks) == (0, 0, 0)


def test_compile_counter_counts_the_same_under_fullgraph():
    def step(x, pos):
        return x * 2 + pos

    def branchy(x):
        if x.sum() > 0:
            return x + 1
        return x - 1

    for fullgraph in (False, True):
        with CompileCounter() as cc:
            compiled = torch.compile(step, backend="eager", fullgraph=fullgraph)
            for pos in range(6):
                compiled(torch.ones(3), torch.tensor(pos))
        assert (cc.compilations, cc.graphs) == (1, 1), fullgraph
        with CompileCounter() as cc:
            compiled = torch.compile(step, backend="eager", fullgraph=fullgraph, dynamic=False)
            for pos in range(5):
                compiled(torch.ones(3), pos)
            assert cc.snapshot().compilations == 5
        assert (cc.compilations, cc.graphs) == (5, 5), fullgraph

    # Under fullgraph=True a graph break is an exception, and nothing was compiled.
    with CompileCounter() as cc:
        compiled = torch.compile(branchy, backend="eager", fullgraph=True)
        with pytest.raises(torch._dynamo.exc.Unsupported):
            compiled(torch.ones(3))
    assert cc.compilations == 0 and cc.graph_breaks == 1
    # The metrics limit torch ships with is back in place.
    from torch._dynamo import utils

    assert utils._compilation_metrics.maxlen == utils.DEFAULT_COMPILATION_METRICS_LIMIT


# -- TransferSpy --------------------------------------------------------------------------


def test_transferspy_records_pins_transfers_and_marks_in_order():
    spy = TransferSpy()
    batches = [spy.label(torch.full((2,), float(i)), i) for i in range(3)]
    device = torch.device("cpu")

    # A correct prefetcher: batch i + 1 is on its way before batch i is handed out.
    def prefetch():
        pending = spy.to_device(spy.pin(batches[0]), device, non_blocking=True)
        for following in batches[1:]:
            ready, pending = pending, spy.to_device(spy.pin(following), device, non_blocking=True)
            yield ready
        yield pending

    seen = []
    for i, batch in enumerate(prefetch()):
        spy.mark(("consumed", i))
        seen.append(batch.tolist())
    assert seen == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    assert spy.order() == [
        ("pin", 0), ("transfer", 0), ("pin", 1), ("transfer", 1), ("mark", ("consumed", 0)),
        ("pin", 2), ("transfer", 2), ("mark", ("consumed", 1)), ("mark", ("consumed", 2)),
    ]  # fmt: skip
    assert spy.unpinned == [] and spy.blocking == [] and len(spy.transfers) == 3

    # Pageable memory, and a blocking copy: both show.
    spy = TransferSpy()
    source = spy.label(torch.ones(2), "a")
    moved = spy.to_device(source, device, non_blocking=True)
    spy.to_device(spy.pin(source), device)
    assert [(e.pinned, e.non_blocking) for e in spy.transfers] == [(False, True), (True, False)]
    assert len(spy.unpinned) == 1 and len(spy.blocking) == 1
    assert spy.label_of(moved) == "a" and moved is not source
    assert "pinned=False" in spy.describe()
