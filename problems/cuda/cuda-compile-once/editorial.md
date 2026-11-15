## The idea

`torch.compile(fn)` does nothing when it is called. The first time the result is
called, Dynamo runs `fn`'s bytecode symbolically, records every tensor operation into
a graph, hands the graph to the backend, and stores the compiled code together with
a list of **guards**: the conditions under which that trace is the right one. Every
later call checks the guards first. If they hold, the compiled code runs; if not,
the function is traced and compiled again and a second entry is stored next to the
first.

What a guard looks at decides everything in this problem:

| argument | guarded | not guarded |
|---|---|---|
| a tensor | shape, dtype, device, `requires_grad` | its values |
| a Python `int`, `float`, `bool`, `str` | **its value** | |
| an object reached through a closure (`model`, `cache`) | identity, the attributes the trace read | tensor values inside it |

A Python number is a constant of the trace: `pos + slots[:T]` with `pos = 5` is
recorded as "add 5", so the trace is only valid while `pos == 5`. The same line with
a 0-d tensor is recorded as "add the second input", valid for every value. That is
the whole difference between one graph and one graph per token, and nothing in the
output shows it.

With the default `dynamic=None`, torch softens the blow: on the first guard failure
it recompiles with the offending size or integer made *symbolic*, so an `int`
position costs two graphs and not sixty-four. That repair has a price (a graph with
symbolic sizes optimizes less, and under CUDA graphs each new size is still a new
recording), and it hides the bug. `dynamic=False` is the honest setting for a decode
step: shapes are meant to be static, so a recompilation is a defect, and it shows
up as one. Likewise `fullgraph=True` turns a graph break, which otherwise only costs
speed, into an error at the first call.

So the job is to make the set of guarded things that reach `step` small and fixed:

- **Positions are tensors.** `torch.tensor(P, device=device)`, incremented in place.
- **Decode blocks are always `(batch, 1)`.** The cache is static (the previous
  problem), so the step never sees the sequence grow.
- **Prompt blocks come in a few lengths.** Padding to a multiple of 16 maps prompts
  of 1 to 16 tokens to one shape, 17 to 32 to the next. Right-padding is safe for a
  causal model with a static cache: the padding is after every real token, so the
  mask keeps it out of their attention; its keys and values land in slots $P$ to
  $L - 1$, and the decode step for slot $s$ overwrites slot $s$ before any query
  attends to it. The only place the padding shows is in the logits of the padded
  rows, so the first token must be read at index $P - 1$, and decoding must start
  at slot $P$.
- **One `torch.compile`, outside every loop.**

## Reference solution (PyTorch)

```python
def make_step(model, cache, backend="inductor", mode=None):
    def step(tokens, pos):
        return model(tokens, pos, cache)

    return torch.compile(step, fullgraph=True, dynamic=False, backend=backend, mode=mode)


def generate(step, prompt, n_new, device, bucket=16):
    batch, n_prompt = prompt.shape
    padded = -(-n_prompt // bucket) * bucket
    tokens = torch.zeros((batch, padded), dtype=prompt.dtype, device=device)
    tokens[:, :n_prompt] = prompt
    out = torch.empty((batch, n_new), dtype=prompt.dtype, device=device)

    logits = step(tokens, torch.tensor(0, device=device))
    token = logits[:, n_prompt - 1].argmax(dim=-1, keepdim=True)
    out[:, 0] = token[:, 0]
    pos = torch.tensor(n_prompt, device=device)
    for i in range(1, n_new):
        token = step(token, pos)[:, 0].argmax(dim=-1, keepdim=True)
        out[:, i] = token[:, 0]
        pos += 1
    return out
```

`n_prompt - 1`, `i`, and the slice bounds are Python ints, and that is fine: they are
used in `generate`, which is not compiled, so nothing guards on them. The rule is not
"no Python ints"; it is "no Python int that changes may cross into the compiled
function". The token goes from one step to the next as a tensor and never visits the
host, and the loop runs `n_new - 1` decode steps: the last token is an output only.

## Common mistakes

- **A Python `int` position.** Same tokens; one graph per decode step under
  `dynamic=False`, and the eighth is the last: Dynamo's `recompile_limit` is 8 per
  function, after which it stops compiling the function (and with `fullgraph=True`
  raises `FailOnRecompileLimitHit`).
- **Prompts at their own length.** One graph per distinct prompt length: a
  compilation on the request path for every length not seen before.
- **Feeding the whole sequence each step.** The shape grows by one token per step;
  it also throws away the cache.
- **A Python `if` on a tensor inside the step** (a bounds check on `pos`, an
  end-of-sequence test). Without `fullgraph=True` Dynamo compiles the code before
  the branch, runs the branch in Python with a host read, and compiles the rest: two
  or more graphs per shape and a synchronization per step.
- **Leaving `dynamic` at its default.** The count stays low because torch
  generalizes on the first failure; the mistake that caused the failure stays in.
- **`torch.compile` inside the loop.** It builds no new graph, because Dynamo's
  cache hangs off the function's code object, which is why this survives so long in
  real code. It still rebuilds the wrapper and repeats the cache lookup and guard
  checks on a fresh entry point at every step, and with a backend that keeps state
  per compiled function it would throw that state away.
- **Reading the first token from `logits[:, -1]`.** That is the prediction after a
  run of padding tokens.
- **Starting the decode at the padded length.** The tokens get positions $L, L + 1,
  \dots$ and attend to the padding in between.
- **Padding everything to the longest length.** One prompt graph, and every short
  prompt pays for a full-length forward pass; buckets are the compromise.
- **Moving the token through the host** (`.item()`, `.tolist()`). The compiler does
  not care, the device does: one synchronization per token.
- **Running the model on the last new token.** A forward pass whose logits nobody
  reads.

## Complexity

One compilation per distinct shape: $1 + {}$(number of buckets in use). Per request,
one forward pass over $L \le P + \text{bucket} - 1$ tokens and `n_new - 1` passes over
one token, each attending to all `max_len` slots of the static cache.

## In the wild

Hugging Face Transformers pairs its `StaticCache` with
`torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)` and documents
padding prompts to a multiple of a fixed length to avoid recompilation.
`TORCH_LOGS=recompiles` prints the failed guard for every recompilation, which is the
first thing to look at when a compiled loop is slow:
`pos == 5` in that log is this problem's first mutant.

What was verified where. On the CPU build of torch 2.15 with `backend="eager"`: the
graph counts in this editorial (2 for the worked example, 4 with `int` positions, 8
and then `FailOnRecompileLimitHit` over a longer loop, one per prompt length without
padding), that `torch.compile` inside the loop adds no graph, and that Dynamo's
`frames.ok` counter stays at 0 under `fullgraph=True` (the judge counts graphs handed
to the backend instead). Not verified: anything about the default backend on a GPU,
including `mode="reduce-overhead"`, which the documentation describes as applying
only to graphs that run entirely on CUDA and do not mutate their inputs; whether it
applies to a step that writes into a static cache is exactly the kind of fact to
check with `TORCH_LOGS` on the first GPU run. The GPU case has never been run.
