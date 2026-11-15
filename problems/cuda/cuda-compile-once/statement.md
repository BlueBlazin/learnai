Wrap a decode step in `torch.compile` and drive a greedy decoding loop through it so
that the compiler runs **once for the decode step and once per padded prompt length**,
however many tokens and requests follow. The model and its static cache are given;
what has to be right is everything the compiler guards on: which arguments are tensors
and which are Python numbers, and which shapes ever reach the compiled function.

## Specification

```python
def make_step(model, cache, backend="inductor", mode=None): ...  # returns step
def generate(step, prompt, n_new, device, bucket=16): ...  # returns tokens
```

**The model** is a causal transformer provided by the judge, called as
`model(tokens, pos, cache)`:

- `tokens`: int64 token ids of shape `(batch, T)`.
- `pos`: the slot of `tokens[:, 0]`. The model accepts a 0-d int64 tensor **or a
  Python int** and computes the same numbers for both.
- `cache`: a static KV cache as in `cuda-static-kv-cache`, with `cache.max_len` slots.
  The model writes the keys and values of the `T` tokens into slots `pos` to
  `pos + T - 1`, in place, and lets the token in slot $s$ attend to slots $0$ to $s$,
  whatever the other slots hold.
- Returns float32 logits of shape `(batch, T, vocab)`; row `[:, i]` is the prediction
  for the token after `tokens[:, i]`.

**`make_step(model, cache, backend, mode)`** returns a function
`step(tokens, pos)` that returns `model(tokens, pos, cache)`, compiled by **one** call
of

```python
torch.compile(..., fullgraph=True, dynamic=False, backend=backend, mode=mode)
```

- `fullgraph=True`: code the compiler cannot trace is an error instead of a silent
  split of the function into pieces (a *graph break*).
- `dynamic=False`: the compiler never generalizes. It records *guards* while tracing
  (the shape and dtype of every tensor argument, the **value** of every Python number
  it used) and compiles a new graph whenever a call fails them.

`torch.compile` is lazy: it returns at once, and the first call of `step` with a new
combination of guarded properties pays for the compilation.

**`generate(step, prompt, n_new, device, bucket)`** decodes greedily.

- `prompt`: int64, shape `(batch, P)` with $P \ge 1$, on `device`. All rows have the
  same length.
- It returns the `n_new` $\ge 1$ new tokens as an int64 tensor of shape
  `(batch, n_new)` on `device`: token $i$ is the argmax of the logits after the
  prompt and tokens $0$ to $i - 1$.
- It calls `step` exactly `n_new` times:
  1. once on the prompt, **padded on the right with zeros** to the next multiple of
     `bucket`, $L = \text{bucket} \cdot \lceil P / \text{bucket} \rceil$, at position
     0. The first new token is the argmax of the logits of the last *real* token,
     index $P - 1$ of the block;
  2. then once for each new token except the last, on a block of shape `(batch, 1)`,
     at positions $P, P + 1, \dots, P + \text{n\_new} - 2$. The logits of the last
     new token are never needed, so the model is not run on it.
- Every position passed to `step` is a **0-d int64 tensor on `device`**.
- Nothing is read on the host: no `.item()`, `.tolist()`, `int(...)`, or `if` on a
  tensor. Every tensor is created with `device=`.
- `prompt` is not modified.

> [!NOTE]
> Padding on the right is safe here because of how the model uses the cache. The
> padding tokens sit after every real token, so no real token attends to them, and
> the slots they fill ($P$ to $L - 1$) are overwritten by the decode steps, one by
> one, before any token attends to them.

**The cost.** With `dynamic=False`, one graph is built per distinct combination of
shapes and Python values that reaches `step`. A correct solution builds **one graph
for the decode shape and one for each distinct padded prompt length**, and none of
them has a graph break. The judge counts graphs, graph breaks, the calls of
`torch.compile`, and the calls of `step`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | `(batch, P)` | int64 |
| `tokens` passed to `step` | `(batch, L)` for the prompt, `(batch, 1)` afterwards | int64 |
| `pos` passed to `step` | `()` | int64 |
| result of `step` | `(batch, T, vocab)` | float32 |
| result of `generate` | `(batch, n_new)` | int64 |

## Example

One request with a prompt of 5 tokens, `n_new = 4`, and the default `bucket = 16`:

```python
step = make_step(model, cache, backend="eager")
generate(step, prompt, 4, device)  # prompt: shape (1, 5)

# the calls of step, in order:
#   step(tokens of shape (1, 16), pos = tensor(0))   prompt + 11 zeros; read logits[:, 4]
#   step(tokens of shape (1, 1),  pos = tensor(5))   new token 0 in, new token 1 out
#   step(tokens of shape (1, 1),  pos = tensor(6))
#   step(tokens of shape (1, 1),  pos = tensor(7))   new token 3 out; not fed back
```

Two shapes reach `step`, `(1, 16)` and `(1, 1)`, so two graphs are built: one by the
first call and one by the second. The third and fourth calls pass every guard of the
second graph, because only the *values* of `tokens` and `pos` have changed, and values
of tensors are not guarded.

Passing the positions as Python ints (`0`, `5`, `6`, `7`) gives the same tokens and
four graphs: the guard of the second graph includes `pos == 5`, the third call fails
it, and so does every call after. A second request with a prompt of 12 tokens reuses
both graphs when it is padded to 16, and costs a third graph when it is not.

## Constraints

- $1 \le \text{batch} \le 3$, $1 \le P \le 40$, $1 \le \text{n\_new} \le 64$, `bucket`
  is 8 or 16, and $L \le \text{cache.max\_len}$, $P + \text{n\_new} \le
  \text{cache.max\_len} = 96$.
- Tokens are compared exactly with greedy decoding that uses neither padding, nor a
  cache, nor `torch.compile` (the logits are not close to a tie). The logits returned
  by `step` are compared with the same computation to a relative tolerance of `1e-4`
  and an absolute tolerance of `1e-4`.
- Limits, checked for one `make_step` and all requests made through it: at most
  $1 + {}$(number of distinct padded prompt lengths) graphs, or just the prompt graph
  when every request has `n_new = 1`; 0 graph breaks; exactly 1 call of
  `torch.compile`, written as `torch.compile(...)`; exactly `n_new` calls of `step`
  per request; 0 host synchronizations in `generate`.
- When one function reaches 8 graphs, Dynamo's `recompile_limit`, `fullgraph=True`
  turns the next recompilation into an exception; the judge reports it as a failure.
- **What runs where.** The cases without a hardware mark run on the CPU with
  `backend="eager"`: Dynamo traces, guards, and recompiles exactly as it does for any
  backend, and no compiler for generated code is needed. One case runs only on a
  machine with an NVIDIA GPU: the default backend with `mode="reduce-overhead"` on
  the CUDA device, with tokens and logits compared against the CPU computation. That
  case was written from the PyTorch 2.15 documentation and could not be run when the
  problem was authored.

## Why this matters

A compiled decode step removes most of the Python and dispatch overhead of a small
model, and with `mode="reduce-overhead"` torch also tries to replay the step as a
CUDA graph (the subject of the next problem). The compilation it costs is seconds to minutes, against a step of
milliseconds. A serving process that passes the position as an `int`, slices the
cache to the current length, or runs each prompt at its own length compiles on the
request path: the first requests take minutes instead of milliseconds, each new
prompt length stalls a worker again, and after eight variants of one function torch
stops compiling it and quietly runs it uncompiled (or, with `fullgraph=True`, raises).
The numbers are right in every one of these cases, so nothing but a latency graph or
a count of compilations shows the mistake. Padding prompts to a few bucket lengths,
keeping every changing number in a tensor, and compiling once at startup is how
engines keep the compiler off the request path.
