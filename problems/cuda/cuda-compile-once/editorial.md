## The idea

Calling `torch.compile(fn)` does no compilation upfront. On the first call to the wrapped function, TorchDynamo (Dynamo) traces `fn`'s bytecode symbolically, records every tensor operation into a graph, hands that graph to the backend, and caches the compiled code alongside a set of **guards**. Guards are the conditions under which that compiled graph remains valid. On every later call, Dynamo checks the guards first. If they pass, the cached graph runs. If any guard fails, Dynamo traces and compiles the function again and stores a new entry next to the old one.

What Dynamo guards on determines the entire design of this problem:

| argument | guarded | not guarded |
|---|---|---|
| a tensor | shape, dtype, device, `requires_grad` | its values |
| a Python `int`, `float`, `bool`, `str` | **its value** | |
| an object reached through a closure (`model`, `cache`) | identity, attributes read during tracing | tensor values inside it |

A Python number becomes a compile-time constant in the trace. Evaluating `pos + slots[:T]` for the cache slot indices with `pos = 5` records "add 5" and adds the guard `pos == 5`. Evaluating the same expression with a 0-d tensor records "add the second input", which stays valid for every value. That single distinction separates compiling once from compiling on every token, even though both produce identical logits.

Under the default `dynamic=None`, PyTorch hides this mistake. On the first guard failure, it recompiles with the offending size or integer marked *symbolic*, so a Python `int` position costs two graphs instead of sixty-four. Symbolic shapes limit backend optimizations and still require a new recording per size under CUDA graphs. Setting `dynamic=False` treats every unexpected recompilation as a bug, and `fullgraph=True` turns silent graph breaks into immediate errors.

To keep the set of guarded properties small and fixed:

- **Pass positions as tensors.** Create `torch.tensor(P, device=device)` and increment it in place.
- **Keep decode blocks at shape `(batch, 1)`.** Because the KV cache is static, the decode step never sees a growing sequence.
- **Bucket prompt lengths.** Right-padding prompts to a multiple of `bucket` (for example, 16) maps lengths 1 through 16 to one shape and 17 through 32 to the next. Causal masking prevents real tokens from attending to padding tokens after them, and decoding overwrites slots $P$ through $L - 1$ (where $L$ is the padded length) before any later query attends to them. You only need to read the first predicted token from `logits[:, P - 1]` and start decoding at position $P$.
- **Call `torch.compile` once, outside every loop.**

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

`n_prompt - 1`, `i`, and the slice indices are Python ints, which is completely fine because `generate` is not compiled and nothing guards on them. The rule is only that changing Python ints must not cross into the compiled `step` function. Each generated token stays on `device` as a `(batch, 1)` tensor without touching the host, and the loop runs `n_new - 1` decode steps because the final token is never fed back into the model.

## Common mistakes

- **Passing `pos` as a Python `int`.** The outputs stay correct, but `dynamic=False` compiles one graph per decode step. At the eighth graph, Dynamo hits `recompile_limit = 8`, stops compiling, and raises `FailOnRecompileLimitHit` under `fullgraph=True`.
- **Running prompts at their unpadded length.** Each unseen prompt length triggers a fresh compilation on the request path.
- **Feeding the full sequence on every decode step.** The input shape grows by one token per step, forcing a recompilation every step and discarding the KV cache.
- **Branching in Python on a tensor inside `step` (such as a bounds check on `pos` or an EOS check).** Without `fullgraph=True`, Dynamo splits the function at the branch, reads the tensor on the host, and compiles the pieces separately. That creates multiple graphs per shape and adds a host synchronization every step.
- **Leaving `dynamic` at its default (`None`).** PyTorch silently generalizes on the first guard failure, which keeps the graph count low while hiding the underlying bug.
- **Calling `torch.compile` inside the loop.** Because Dynamo caches compiled graphs on the function's code object, this does not increase the graph count and often goes unnoticed. However, it rebuilds the wrapper and repeats cache lookup and guard checks on every step, and it discards per-callable state in backends that maintain any.
- **Reading the first token from `logits[:, -1]`.** Index `-1` is the prediction after the padding tokens rather than the last real prompt token.
- **Starting decoding at the padded length `L`.** New tokens land at positions $L, L + 1, \dots$ and attend to the padding tokens stored in slots $P$ through $L - 1$.
- **Padding every prompt to `max_len`.** This uses only one prompt graph, but forces every short prompt to pay for a full-length forward pass. Bucketing balances graph count against wasted compute.
- **Moving tokens through the host (`.item()`, `.tolist()`).** The compiler does not complain, but every token triggers a device-to-host synchronization.
- **Running `step` on the last generated token.** This wastes a full forward pass whose logits are never used.

## Complexity

Compilation runs once per distinct input shape, for a total of $1 + (\text{number of buckets in use})$ graphs. Each request performs one prefill pass over $L \le P + \text{bucket} - 1$ tokens and `n_new - 1` single-token decode passes, with each pass attending over all `max_len` slots of the static cache.

## In the wild

Hugging Face Transformers pairs `StaticCache` with `torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)` and recommends padding prompts to fixed multiples to prevent recompilation. When a compiled loop runs slowly, setting `TORCH_LOGS=recompiles` prints the exact guard that failed on every recompilation. Seeing `pos == 5` in that log points straight to a Python `int` position.

**Verification notes.** On CPU with PyTorch 2.15 and `backend="eager"`, we verified the exact graph counts in this editorial (2 for the worked example, 4 with `int` positions, 8 followed by `FailOnRecompileLimitHit` on longer loops, and one per prompt length without padding), confirmed that calling `torch.compile` inside the loop adds no graphs, and noted that Dynamo's `frames.ok` counter stays at 0 under `fullgraph=True` (so the judge counts graphs passed to the backend instead). The GPU case with the default backend and `mode="reduce-overhead"` has not been run. PyTorch documents `mode="reduce-overhead"` as applying only to pure-CUDA graphs that do not mutate their inputs, so verifying its behavior with an in-place static cache via `TORCH_LOGS` is a good first step on GPU hardware.
