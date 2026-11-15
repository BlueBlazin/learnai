Capture a step function into a CUDA graph once and replay it on every call. A CUDA
graph records the kernels a piece of code launches together with the memory addresses
they read and write, and a replay launches them again on those addresses without
running a line of Python. Everything about using one correctly follows from that
sentence: the graph does not know your variables, only memory.

## Specification

```python
class GraphedStep:
    def __init__(self, step_fn, example_inputs, graph, warmup=3): ...
    def __call__(self, *inputs): ...
```

**The constructor** prepares the graph.

- `step_fn(*tensors)`: the function to capture. It takes one or more tensors and
  returns a tensor or a tuple of tensors. It may write into state of its own in place
  (a KV cache). Its output shapes depend only on its input shapes.
- `example_inputs`: a sequence of tensors, one per argument of `step_fn`. They fix
  the shape, dtype, and device of each argument, and their values are valid inputs.
- `graph`: an empty `torch.cuda.CUDAGraph`, or on the CPU the judge's stand-in with
  the same methods.
- `warmup`: a Python int $\ge 1$.

It does three things, in this order:

1. **Placeholders.** For each example it allocates a new tensor with the same shape,
   dtype, device, and values (the *static inputs*). The caller's example tensors are
   never written to and never passed to `step_fn`.
2. **Warm-up.** It calls `step_fn` on the static inputs `warmup` times, outside any
   capture.
3. **Capture.** It calls `step_fn` on the static inputs exactly once more inside
   `with torch.cuda.graph(graph):` and keeps what the call returns (the *static
   outputs*). Nothing else happens inside the block: no allocation of placeholders,
   and nothing that reads a tensor's value on the host (`.item()`, `bool(...)`,
   `print`), which a capturing stream does not permit.

`step_fn` therefore runs `warmup + 1` times in the constructor, each time for real
(its in-place writes happen), and never again.

**`step(*inputs)`** runs the step on new inputs:

1. If the number of inputs differs from the number of examples, or any input's shape
   differs from its example's shape, it raises `ValueError` and replays nothing.
2. It copies every input **into** its static input, in place.
3. It calls `graph.replay()`.
4. It returns **copies** of the static outputs: a tensor if `step_fn` returned a
   tensor, a tuple of tensors if it returned a tuple. A result stays valid after any
   number of later calls.

The inputs passed by the caller are not modified, and may be reused or overwritten
by the caller between calls.

![One call of the graphed step. The graph reads the static input and writes the static output at the addresses it recorded, so new values go in with `copy_` and results come out with `clone`.](assets/capture-replay.svg)

> [!NOTE]
> The starter contains `side_stream(device)`, a context manager that runs its block
> on a CUDA stream of its own and does nothing on other devices. PyTorch's CUDA notes
> require the warm-up to run on a side stream, so wrap the warm-up calls in it. (The
> capture gets a side stream from `torch.cuda.graph` itself.)

## Shapes

| name | shape | dtype |
|---|---|---|
| `example_inputs[i]` | any, fixed for the lifetime of the object | any |
| `inputs[i]` | the shape of `example_inputs[i]` | the dtype of `example_inputs[i]` |
| result | the shapes `step_fn` returns for the examples | as returned by `step_fn` |

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

The first call copies `[10, 20, 30]` into the placeholder and replays; the two
recorded kernels (a multiplication and an addition) write `[21, 41, 61]` into the
static output, and `a` is a copy of it. The second call overwrites the static output
with `[1, 1, 3]`.

- Had the call rebound the placeholder (`self.static_input = x`) instead of copying
  into it, both replays would have read `[1, 2, 3]` and returned `[3, 5, 7]`.
- Had it returned the static output itself, `a` and `b` would be the same tensor,
  and `a` would read `[1, 1, 3]` after the second call.

## Constraints

- One to three inputs of up to four dimensions; float32 and int64 tensors, including
  0-d tensors. $1 \le \text{warmup} \le 5$.
- Floating-point outputs are compared with `step_fn` run eagerly on the same inputs
  to a relative tolerance of `1e-4` and an absolute tolerance of `1e-6` (`1e-4` for
  the logits of the decode step); integer outputs exactly. Every result is compared
  again after all later calls.
- The judge counts: calls of `step_fn` (exactly `warmup` before the capture, exactly
  1 during it, 0 afterwards), captures (exactly 1), and replays (exactly 1 per
  successful call, 0 for a call that raises).
- **What runs where.** The cases without a hardware mark run on the CPU with the
  kit's `FakeCudaGraph`, which records the tensor operations run between
  `capture_begin()` and `capture_end()` and reruns them on the same tensor objects
  at `replay()`; `torch.cuda.graph(graph)` accepts it while the judge runs your
  code. It does not imitate streams or memory pools, so `side_stream` is not judged
  there. Two cases run only on a machine with an NVIDIA GPU, with a real
  `torch.cuda.CUDAGraph`: a two-layer network compared with the CPU, and a decode
  step over a static KV cache. They were written from the PyTorch 2.15 documentation
  and could not be run when the problem was authored.

## Why this matters

A decode step of a small or medium model on a GPU is hundreds of tiny kernels, and
for each one PyTorch spends tens of microseconds in Python, the dispatcher, and the
driver before the GPU does a few microseconds of work. A CUDA graph submits the whole
step with one call, which is where `torch.compile(mode="reduce-overhead")` and the
graph modes of serving engines get their speedup at small batch sizes. The price is a
contract with no safety net. A replay cannot fail on a wrong input, because it has
no inputs: a tensor that was rebound instead of copied into gives the previous
request's answer, at full speed. An output that was not copied changes after it was
returned, typically one step later, inside someone else's sampling code. And a
capture that includes first-call initialization, an allocation of its own inputs, or
a host read either fails at capture time or, worse, replays the wrong thing forever.
