## The idea

Between `capture_begin()` and `capture_end()`, CUDA does not execute kernels on the stream normally. Instead, it records each kernel launch together with its arguments, which are raw GPU memory addresses. Calling `replay()` submits that entire recorded sequence at once with no Python execution, no dispatcher overhead, no memory allocation, and no shape computation. That mechanism explains every constraint of using CUDA graphs:

- **Inputs must be copied in place.** A replay takes no arguments and always reads from the exact addresses captured during recording. To feed new data into the graph, overwrite the captured memory with `static_input.copy_(x)`. Rebinding `static_input = x` only changes a Python reference that the graph never sees.
- **Outputs must be cloned out.** Every replay writes into the same output memory addresses. To preserve a result past the next replay, copy it out with `static_output.clone()`.
- **Shapes, control flow, and host syncs are frozen.** Because kernels and addresses are fixed at capture time, shapes cannot change and data-dependent branches cannot run. Calling `.item()` during capture raises an error because nothing has executed yet to produce a value.

The constructor's three-step order follows directly from these rules.

**Allocate placeholders before capture.** If you clone `example_inputs` inside the capture block, the copy kernels themselves get recorded. Every replay then starts by overwriting the placeholders with the original example values, erasing whatever `copy_` just wrote.

**Warm up before capture.** The first execution of GPU code performs one-time setup such as lazy library initialization, kernel autotuning, and allocator pool growth. Running `step_fn` eagerly `warmup` times flushes that work out so it does not fail capture or get replayed on every call. PyTorch's CUDA graph rules require warm-up to run on a side stream, which the provided `side_stream` context manager handles (`torch.cuda.graph` handles the side stream for capture).

**Capture once.** Recording still executes `step_fn`, so a stateful function runs `warmup + 1` times during construction. When capturing a decode step over a KV cache, the example token is written `warmup + 1` times into the example slot, so you should build the graphed step before prefill or point it at a slot that real decoding will overwrite.

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

`GraphedStep` does not store `step_fn` because the Python function is never called again after capture. Checking shapes explicitly before calling `copy_` is essential because `copy_` broadcasts silently. Without that check, a `(1, n)` input would fill every row of an `(m, n)` placeholder instead of raising `ValueError`.

Using `with torch.cuda.graph(graph):` is safer than calling `capture_begin()` and `capture_end()` manually. Raw capture must occur on a non-default stream, and the context manager sets up that stream while synchronizing, running garbage collection, and emptying the allocator cache before capture starts.

## Common mistakes

- **Rebinding placeholders (`self.static_inputs = tuple(inputs)`).** The graph still reads the original placeholder addresses, so every replay returns the result for `example_inputs`.
- **Copying only a subset of inputs.** The uncopied placeholders keep stale values from the previous call.
- **Returning static outputs directly.** Results look right until the caller holds onto an output across a second call, at which point the next replay overwrites it.
- **Cloning before calling `replay()`.** Each call returns the previous call's output instead of the current one.
- **Skipping warm-up, putting warm-up inside capture, or hard-coding the warm-up count.** Putting warm-up inside capture records `warmup + 1` copies of `step_fn` into the graph, causing every replay to run all of them.
- **Allocating placeholders inside capture.** The copy from `example_inputs` gets recorded, so every replay resets the inputs before running `step_fn`.
- **Using `example_inputs` directly as placeholders.** Every call overwrites the caller's tensors, and the caller can mutate the graph's input memory unexpectedly.
- **Reading a tensor on the host inside capture (such as a NaN check, `print`, or `assert`).** On CUDA, this fails with `RuntimeError: CUDA error: operation not permitted when stream is capturing`.
- **Calling `step_fn` instead of `graph.replay()`.** This produces the right numbers but completely defeats the graph speedup.
- **Omitting the shape check.** Because `copy_` broadcasts, a mismatched shape like `(1, n)` silently overwrites an `(m, n)` placeholder instead of raising `ValueError`.

## Complexity

The constructor runs `step_fn` `warmup + 1` times. Each call to `GraphedStep` performs one in-place copy per input, one graph launch, and one clone per output. Memory for the static inputs, static outputs, and every intermediate tensor from the captured run stays allocated for the lifetime of the graph.

## In the wild

`torch.cuda.make_graphed_callables` follows this exact pattern with autograd support. It allocates static inputs, runs three warm-up iterations on a side stream, captures the graph, and returns a callable that copies inputs in, replays, and returns the static outputs. `torch.compile(mode="reduce-overhead")` captures a CUDA graph per compiled graph and input shape, and it raises a runtime error if user code reads an output tensor after a later replay has overwritten it (catching the missing-`clone` bug automatically). Capturing a decode step also depends on the static KV cache from `cuda-static-kv-cache` (since `torch.cat` allocates a new address every step) and tensor positions from `cuda-compile-once`.

**Verification notes.** On CPU with the kit's `FakeCudaGraph`, we verified every failure mode in "Common mistakes" except the warm-up behaviors, which `FakeCudaGraph` cannot reproduce and the judge verifies by tracking when `step_fn` is called. From the PyTorch 2.15 documentation (not run on GPU): warm-up requires a side stream, host reads and dynamic shapes fail during capture, replays reuse fixed memory addresses, and `torch.cuda.graph`, `torch.cuda.Stream`, and `torch.cuda.stream` follow the API used by the reference and `side_stream`. The GPU tests have not been run.
