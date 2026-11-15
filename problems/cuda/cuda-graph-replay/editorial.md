## The idea

Between `capture_begin()` and `capture_end()` CUDA does not execute the kernels a
stream is given; it records them: which kernel, with which arguments, and the
arguments of a kernel are raw addresses. `replay()` submits the whole recording at
once. No Python runs, no dispatcher, no allocator, no shape computation. That is the
speedup, and it is also the complete list of things a replay cannot do:

- It cannot take arguments. It reads whatever is, at that moment, in the memory the
  captured kernels read. To give it new input you write into that memory:
  `static_input.copy_(x)`. Assigning `static_input = x` changes a Python name; the
  graph has never heard of it.
- It cannot return anything. It writes to the memory the captured kernels wrote to,
  the same memory at every replay. To keep a result past the next replay you copy it
  out: `static_output.clone()`.
- It cannot change shape, branch on data, or talk to the host. Kernels and addresses
  were fixed at capture time; a `.item()` during capture is an error because there
  is nothing executed yet to read.

The constructor's order follows from the same picture.

**Placeholders first, outside the capture.** If `x.clone()` runs inside the capture
block, the copy kernel is part of the recording: every replay starts by copying the
example values into the placeholder again, over whatever `copy_` just wrote there.

**Then warm-up.** The first run of anything on a GPU does work that later runs do
not: lazy initialization of libraries, picking algorithms, growing the allocator's
pools. Captured, that work would be replayed forever, or would fail the capture.
A few eager runs on the placeholders flush it out. PyTorch's CUDA notes add that
the warm-up must run on a side stream; `torch.cuda.graph` supplies one for the
capture, and the provided `side_stream` does it for the warm-up.

**Then one captured run.** It executes for real (the recording is made by running
the function), so a stateful `step_fn` has now run `warmup + 1` times. With a KV
cache that means the example token was written `warmup + 1` times into the example
position; build the graphed step before the prefill, or with an example position
that the real sequence overwrites.

## Reference solution (PyTorch)

```python
class GraphedStep:
    def __init__(self, step_fn, example_inputs, graph, warmup=3):
        self.graph = graph
        self.static_inputs = tuple(x.clone() for x in example_inputs)
        with side_stream(self.static_inputs[0].device):
            for _ in range(warmup):
                step_fn(*self.static_inputs)
        with torch.cuda.graph(graph):
            self.static_outputs = step_fn(*self.static_inputs)

    def __call__(self, *inputs):
        if len(inputs) != len(self.static_inputs):
            raise ValueError(f"expected {len(self.static_inputs)} inputs, got {len(inputs)}")
        for static, new in zip(self.static_inputs, inputs, strict=True):
            if new.shape != static.shape:
                raise ValueError(f"expected shape {tuple(static.shape)}, got {tuple(new.shape)}")
        for static, new in zip(self.static_inputs, inputs, strict=True):
            static.copy_(new)
        self.graph.replay()
        if isinstance(self.static_outputs, torch.Tensor):
            return self.static_outputs.clone()
        return tuple(out.clone() for out in self.static_outputs)
```

The object does not keep `step_fn`: after the capture, the function's Python is
dead weight. The shape check is done by hand and before any copy, because `copy_`
broadcasts: a `(1, n)` input would fill every row of an `(m, n)` placeholder without
a word.

`torch.cuda.graph(graph)` is preferred over calling `capture_begin()` and
`capture_end()` yourself. PyTorch's notes say a raw capture must be on a non-default
stream; the context manager sets one, and its source also synchronizes, collects
garbage, and empties the allocator's cache before the capture begins.

## Common mistakes

- **Rebinding the placeholder** (`self.static_inputs = tuple(inputs)`). Every replay
  returns the result for the example inputs.
- **Copying only some of the inputs.** The rest keep the previous call's values.
- **Returning the static outputs.** Right until the caller keeps one across a call.
- **Cloning before the replay.** Each call returns the previous call's result.
- **No warm-up**, or the warm-up inside the capture block (the graph then contains
  `warmup + 1` copies of the step and every replay runs them all), or a hard-coded
  number of warm-up runs.
- **Placeholders allocated inside the capture.** The fill is recorded, and replays
  reset the inputs.
- **Using the example tensors as placeholders.** Every call scribbles over the
  caller's tensors, and the caller can change the graph's inputs behind its back.
- **A host read inside the capture** (a NaN check, a `print`, an `assert` on a
  tensor). `RuntimeError: CUDA error: operation not permitted when stream is
  capturing`.
- **Calling `step_fn` instead of `replay()`.** Correct, and exactly as slow as
  before.
- **No shape check.** `copy_` broadcasts a wrong shape instead of failing.

## Complexity

The constructor runs the step `warmup + 1` times. A call costs one copy per input,
one graph launch, and one copy per output; the memory of the static inputs and
outputs, and of every intermediate tensor of the captured run, stays allocated for
as long as the graph lives.

## In the wild

`torch.cuda.make_graphed_callables` is this class with autograd support: it
allocates static inputs, warms up (three iterations by default, on a side stream),
captures, and returns a callable that copies, replays, and hands back the static
outputs. `torch.compile(mode="reduce-overhead")` records a CUDA graph per compiled
graph and input shape, and raises an error when code touches an output that a later
run has overwritten, which is the missing-`clone` mistake caught at run time.
A graphed decode step needs the static cache of `cuda-static-kv-cache` (a cache
grown with `torch.cat` has a new address at every step) and positions passed as
tensors, as in `cuda-compile-once`.

What was verified where. On the CPU, with the kit's `FakeCudaGraph`: every
consequence listed under "Common mistakes" except the two about warm-up, which the
stand-in cannot produce and the judge checks by watching when `step_fn` is called.
From the PyTorch 2.15 documentation, not run: that warm-up belongs on a side stream,
that host reads and dynamic shapes are prohibited during capture, that replays use
the same memory addresses, and the `torch.cuda.graph`, `torch.cuda.Stream`, and
`torch.cuda.stream` calls used by the reference and by `side_stream`. The GPU cases
have never been run.
