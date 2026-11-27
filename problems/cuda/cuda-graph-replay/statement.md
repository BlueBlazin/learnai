Capture a step function into a CUDA graph once and replay it on every subsequent call. A CUDA graph records the kernels launched by a block of code along with the exact memory addresses they read and write. Replaying the graph launches those same kernels on those same addresses without running any Python code, so you must move new inputs into the captured memory before replay and copy outputs out afterward.

## Specification

```python
class GraphedStep:
    def __init__(self, step_fn, example_inputs, graph, warmup=3): ...
    def __call__(self, *inputs): ...
```

**`__init__(self, step_fn, example_inputs, graph, warmup=3)`** prepares and captures the graph.

- `step_fn(*tensors)` is the callable to capture. It takes one or more tensors and returns either a single tensor or a tuple of tensors. It may update its own state in place (such as a KV cache), and its output shapes depend only on its input shapes.
- `example_inputs` is a sequence of valid input tensors, one per argument of `step_fn`. They fix the shape, dtype, and device for all future calls.
- `graph` is an empty `torch.cuda.CUDAGraph` (or, on CPU, the judge's stand-in with the same methods).
- `warmup` is a Python `int` with $\text{warmup} \ge 1$.

The constructor performs three steps in order:

1. **Allocate placeholders.** For each tensor in `example_inputs`, allocate a new tensor with the same shape, dtype, device, and values (the *static inputs*). Never modify the caller's `example_inputs` or pass them directly to `step_fn`.
2. **Warm up.** Call `step_fn` on the static inputs `warmup` times outside any capture.
3. **Capture.** Call `step_fn` on the static inputs once more inside `with torch.cuda.graph(graph):` and store its return value (the *static outputs*). Keep everything else out of the capture block. Do not allocate placeholders inside it, and do not read tensor values on the host (`.item()`, `bool(...)`, `print`), which a capturing stream forbids.

In total, `step_fn` runs `warmup + 1` times in the constructor (executing its in-place writes each time) and never runs again afterward.

**`__call__(self, *inputs)`** runs the captured step on new inputs:

1. If the number of inputs does not match `example_inputs`, or if any input's shape differs from its corresponding example's shape, raise `ValueError` without replaying the graph.
2. Copy each input **into** its corresponding static input in place.
3. Call `graph.replay()`.
4. Return **copies** of the static outputs: a tensor if `step_fn` returned a tensor, or a tuple of tensors if it returned a tuple. Returned tensors must remain unchanged across any number of later calls.

Never modify the caller's `inputs`, and assume the caller may reuse or overwrite them between calls.

![One call of the graphed step. New values go into the static input with `copy_`, and results leave the static output with `clone`.](assets/capture-replay.svg)

> [!NOTE]
> The starter code provides `side_stream(device)`, a context manager that runs its block on a dedicated CUDA stream and acts as a no-op on other devices. PyTorch's CUDA graph guidelines require warm-up runs to execute on a side stream, so wrap the warm-up loop in `side_stream(device)`. (`torch.cuda.graph` already creates a side stream for the capture itself.)

## Shapes

| name | shape | dtype |
|---|---|---|
| `example_inputs[i]` | any, fixed for the lifetime of the object | any |
| `inputs[i]` | same as `example_inputs[i]` | same as `example_inputs[i]` |
| result | shapes returned by `step_fn` on `example_inputs` | dtypes returned by `step_fn` |

## Example

```python
graph = torch.cuda.CUDAGraph()  # the judge's stand-in on the CPU
step = GraphedStep(lambda x: x * 2 + 1, (torch.tensor([1.0, 2.0, 3.0]),), graph)
# step_fn has run 4 times on a placeholder holding [1, 2, 3]: 3 warm-up calls and
# the capture. The static output holds [3, 5, 7].

a = step(torch.tensor([10.0, 20.0, 30.0]))  # [21.0, 41.0, 61.0]
b = step(torch.tensor([0.0, 0.0, 1.0]))  # [1.0, 1.0, 3.0]
a  # still [21.0, 41.0, 61.0]
```

The first call copies `[10, 20, 30]` into the static input and replays the graph. The two recorded kernels (multiplication and addition) write `[21, 41, 61]` into the static output, and `a` receives a copy of that tensor. The second call overwrites the static output with `[1, 1, 3]`, leaving `a` intact.

- If `__call__` rebound the placeholder (`self.static_input = x`) instead of copying into it, both replays would still read `[1, 2, 3]` and return `[3, 5, 7]`.
- If `__call__` returned the static output directly instead of cloning it, `a` and `b` would reference the same memory and `a` would change to `[1, 1, 3]` after the second call.

## Constraints

- 1 to 3 inputs, each with up to 4 dimensions, using `float32` and `int64` tensors (including 0-d tensors).
- $1 \le \text{warmup} \le 5$.
- Floating-point outputs are compared against running `step_fn` eagerly on the same inputs with `rtol = 1e-4` and `atol = 1e-6` (`atol = 1e-4` for decode-step logits). Integer outputs are compared exactly. Every returned output is checked again after all subsequent calls complete.
- The judge verifies exact counts: `warmup` calls to `step_fn` before capture, 1 call during capture, 0 calls afterward, 1 capture total, 1 replay per valid call, and 0 replays when `ValueError` is raised.
- **CPU and GPU tests.** Tests without a hardware mark run on CPU using the kit's `FakeCudaGraph`, which records tensor operations between `capture_begin()` and `capture_end()` and reruns them on the same tensor objects during `replay()`. While the judge runs your code, `torch.cuda.graph(graph)` accepts `FakeCudaGraph`. Because `FakeCudaGraph` does not simulate streams or memory pools, `side_stream` is not checked on CPU. Two hardware-marked tests run on an NVIDIA GPU with a real `torch.cuda.CUDAGraph`: a two-layer network compared against CPU and a decode step over a static KV cache. (Those GPU tests were written from the PyTorch 2.15 documentation and were not run when this problem was authored.)

## Why this matters

A decode step for a small or medium model launches hundreds of tiny GPU kernels. For each kernel, PyTorch spends tens of microseconds in Python, the dispatcher, and the CUDA driver before the GPU executes a few microseconds of math. Replaying a CUDA graph submits the entire step in one launch, which is where `torch.compile(mode="reduce-overhead")` and serving-engine graph modes get their low-batch speedup.

The trade-off is a strict memory contract with no runtime guardrails. Because a replay takes no arguments, rebinding a Python variable instead of copying into the static buffer silently returns stale results at full speed. Returning a static output without copying lets the next replay overwrite it while downstream sampling code is still reading it. Capturing one-time initialization, input allocation, or a host read either crashes during capture or replays the wrong work on every call.
