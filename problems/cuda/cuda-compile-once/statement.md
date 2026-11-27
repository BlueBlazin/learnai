Wrap a decode step in `torch.compile` and run a greedy decoding loop through it so that the compiler traces the decode step once and each padded prompt length once, no matter how many tokens or requests follow. The model and its static KV cache are provided. Your job is to control what the compiler guards on, including which arguments are tensors versus Python numbers and which shapes reach the compiled function.

## Specification

```python
def make_step(model, cache, backend="inductor", mode=None): ...  # returns step
def generate(step, prompt, n_new, device, bucket=16): ...  # returns tokens
```

**The model** is a causal transformer provided by the judge and called as `model(tokens, pos, cache)`.

- `tokens` holds token IDs for a batch of size `batch` and sequence length `T`.
- `pos` is the cache slot of `tokens[:, 0]`. The model accepts either a 0-d `int64` tensor or a Python `int` and produces the same output for both.
- `cache` is a static KV cache (as in `cuda-static-kv-cache`) with `cache.max_len` slots. The model writes the keys and values of the `T` tokens into slots `pos` through `pos + T - 1` in place. Each token in slot $s$ attends only to slots $0$ through $s$, regardless of what later slots hold.
- The model returns logits of shape `(batch, T, vocab)`, where `vocab` is the vocabulary size. Row `[:, i]` predicts the token following `tokens[:, i]`.

**`make_step(model, cache, backend, mode)`** returns a function `step(tokens, pos)` that evaluates `model(tokens, pos, cache)`. It must compile `step` with **one** call to:

```python
torch.compile(..., fullgraph=True, dynamic=False, backend=backend, mode=mode)
```

- With `fullgraph=True`, any code the compiler cannot trace raises an error instead of silently splitting the function into pieces (a *graph break*).
- With `dynamic=False`, the compiler never generalizes shapes or scalar values. While tracing, it records *guards* on the shape and dtype of every tensor argument and on the **exact value** of every Python number it reads. Any later call that fails a guard triggers a new compilation.

`torch.compile` is lazy. Calling it returns immediately, and the first call to `step` with a new combination of guarded properties pays the compilation cost.

**`generate(step, prompt, n_new, device, bucket)`** runs greedy decoding on `device` for a `prompt` of length $P \ge 1$ where all rows in the batch share the same length.

- It returns the `n_new` $\ge 1$ generated tokens on `device`. Token $i$ is the argmax of the logits produced after the prompt and tokens $0$ through $i - 1$.
- It calls `step` exactly `n_new` times:
  1. Once on the prompt, **right-padded with zeros** to the next multiple of `bucket`, $L = \text{bucket} \cdot \lceil P / \text{bucket} \rceil$, at position 0. The first new token is the argmax of the logits at index $P - 1$, the last real prompt token.
  2. Once for each new token except the last, passing a block of shape `(batch, 1)` at positions $P, P + 1, \dots, P + \text{n\_new} - 2$. Because the logits of the last new token are never needed, the model is not run on it.
- Every position passed to `step` must be a **0-d `int64` tensor on `device`**.
- Do not read tensor values on the host (no `.item()`, `.tolist()`, `int(...)`, or Python `if` on a tensor). Create every new tensor explicitly with `device=device`.
- Do not modify `prompt`.

> [!NOTE]
> Right-padding is safe because of how the model uses the static cache. The padding tokens sit after all real tokens, so causal attention prevents any real token from attending to them. Their cache slots ($P$ through $L - 1$) are overwritten one by one during decoding before any later token attends to them.

**Compilation budget.** With `dynamic=False`, each distinct combination of tensor shapes and Python values that reaches `step` builds a new graph. A correct solution builds **one graph for the decode shape and one for each distinct padded prompt length**, with zero graph breaks. The judge counts compiled graphs, graph breaks, calls to `torch.compile`, and calls to `step`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | `(batch, P)` | int64 |
| `tokens` passed to `step` | `(batch, L)` for the prompt, `(batch, 1)` afterwards | int64 |
| `pos` passed to `step` | `()` | int64 |
| result of `step` | `(batch, T, vocab)` | float32 |
| result of `generate` | `(batch, n_new)` | int64 |

## Example

Consider a single request with a prompt of 5 tokens, `n_new = 4`, and the default `bucket = 16`:

```python
step = make_step(model, cache, backend="eager")
generate(step, prompt, 4, device)  # prompt: shape (1, 5)

# the calls of step, in order:
#   step(tokens of shape (1, 16), pos = tensor(0))   prompt + 11 zeros; read logits[:, 4]
#   step(tokens of shape (1, 1),  pos = tensor(5))   new token 0 in, new token 1 out
#   step(tokens of shape (1, 1),  pos = tensor(6))
#   step(tokens of shape (1, 1),  pos = tensor(7))   new token 3 out; not fed back
```

Only two shapes reach `step`, `(1, 16)` and `(1, 1)`, so only two graphs are built. The first call builds the prompt graph and the second builds the decode graph. The third and fourth calls pass every guard on the second graph because tensor values are not guarded, so changing the contents of `tokens` and `pos` does not trigger recompilation.

Passing positions as Python ints (`0`, `5`, `6`, `7`) produces the same tokens but builds four graphs. The second graph guards on `pos == 5`, so the third call and every call after it fail the guard and recompile. Similarly, a second request with a 12-token prompt reuses both graphs when padded to 16, whereas an unpadded prompt would build a third graph.

## Constraints

- $1 \le \text{batch} \le 3$, $1 \le P \le 40$, $1 \le \text{n\_new} \le 64$, and `bucket` is 8 or 16.
- $L \le \text{cache.max\_len}$ and $P + \text{n\_new} \le \text{cache.max\_len} = 96$.
- Generated tokens must match unpadded, uncached, uncompiled greedy decoding exactly (the logits are never close to a tie). Logits returned by `step` are checked against the same reference with `rtol = 1e-4` and `atol = 1e-4`.
- Across all requests made through a single `make_step`, the judge enforces:
  - at most $1 + (\text{number of distinct padded prompt lengths})$ compiled graphs, or just the prompt graph if every request has `n_new = 1`,
  - 0 graph breaks,
  - exactly 1 call to `torch.compile`, written as `torch.compile(...)`,
  - exactly `n_new` calls to `step` per request,
  - 0 host synchronizations in `generate`.
- TorchDynamo (the tracer inside `torch.compile`) has a default `recompile_limit` of 8 graphs per function. With `fullgraph=True`, hitting a ninth compilation raises an exception, which the judge reports as a failure.
- **CPU and GPU tests.** Tests without a hardware mark run on CPU with `backend="eager"`, where TorchDynamo still traces, guards, and recompiles identically without needing code generation. One hardware-marked test runs on an NVIDIA GPU using the default backend with `mode="reduce-overhead"` and compares its tokens and logits against the CPU reference. (That GPU test was written from the PyTorch 2.15 documentation and was not run when this problem was authored.)

## Why this matters

Compiling a decode step removes most of the Python and dispatch overhead for small models. With `mode="reduce-overhead"`, PyTorch also tries to capture and replay the step as a CUDA graph (covered in the next problem). Compilation takes seconds to minutes, while a decode step takes milliseconds.

If a serving engine passes the position as a Python `int`, slices the cache to the current length, or runs each prompt at its raw length, it triggers compilation on the request path. The first requests stall for minutes, every unseen prompt length stalls a worker again, and after eight variants PyTorch either stops compiling and falls back to eager mode or raises under `fullgraph=True`. Because the numerical outputs stay completely correct, only latency monitoring or compilation counts reveal the bug. Padding prompts into a few length buckets, keeping changing scalars in tensors, and compiling once at startup keep the compiler off the request path.
