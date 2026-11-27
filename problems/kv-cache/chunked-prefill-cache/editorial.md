## The idea

In a causal transformer, token $i$ attends only to tokens $0 \dots i$, and the KV
cache stores everything later tokens need from earlier ones. As long as each call
continues from the previous call's cache, every token sees the exact same context
whether ten tokens arrive in one call or as `4 + 4 + 2`.

What changes is peak memory. A call over $T$ new tokens with $L$ cached tokens builds
activations for $T$ positions and attention scores of shape $T \times (L + T)$.
Capping $T$ at `chunk_size` bounds that peak while keeping the total computation the
same.

The loop maintains one invariant: *before each call, the cache covers every token
before the current chunk*. Each call then receives only the new chunk, so every token
passes through the model once and the final cache covers the entire prompt.
Because `logits` has one row per token in that call, the row scoring the first token
after the prompt is the last row of the final call. All earlier rows predict tokens
already present in the prompt.

## Reference solution (PyTorch)

```python
def prefill(lm, prompt, chunk_size):
    cache = None
    for start in range(0, len(prompt), chunk_size):
        logits, cache = lm(prompt[start : start + chunk_size], cache)
    return logits[-1], cache
```

Let $P$ be `len(prompt)` and $c$ be `chunk_size`. The loop `range(0, P, c)` produces
$\lceil P / c \rceil$ start indices and never reaches or exceeds $P$, so no chunk is
empty. Python slicing automatically clips the final chunk to the remaining tokens. The
JAX and MLX implementations are identical because the function only slices token lists
and threads the cache through `lm`.

## Common mistakes

- **Re-feeding the prefix.** Calling `lm(prompt[:end])` for a growing `end` produces
  the right logits, so the bug is invisible to a numeric check. However, the model processes
  $c + 2c + 3c + \dots$ tokens (436 instead of 100 for a 100-token prompt with
  `chunk_size=16`), and each call grows with the prefix, defeating the memory cap.
- **Dropping the cache.** Calling `lm(chunk)` without passing `cache` runs every chunk
  as the start of a fresh sequence. The output has the right shape and plausible
  values, but it answers a different, shorter prompt.
- **Making a trailing empty call.** Using a bound like `len(prompt) + 1` or a `while`
  condition `start <= len(prompt)` passes whenever $P$ is not a multiple of $c$ (such
  as $P = 10, c = 4$). When $P$ is an exact multiple of $c$, the final slice is empty
  and `lm` rejects it.
- **Returning the wrong logits row.** Returning the first chunk's logits or the entire
  `(T, vocab_size)` tensor from the last chunk fails whenever there are multiple
  chunks or the last chunk has more than one token.
- **Using the wrong chunk layout.** Rounding the chunk count down and appending the
  remainder to the last chunk puts up to $2c - 1$ tokens in one call, breaking the
  `chunk_size` limit. Splitting from the end (short chunk first) respects the cap but
  violates the specified layout.
- **Ignoring `chunk_size`.** Feeding one token per call gives the right numbers but
  makes $P$ sequential calls instead of $\lceil P / c \rceil$ batched calls.

## Complexity

Chunked prefill performs $\lceil P / c \rceil$ forward calls and processes $P$ tokens.
Total attention work remains $O(P^2)$, identical to a single-call prefill, while peak
attention score memory per layer drops from $O(P^2)$ to $O(c \cdot P)$.

## In the wild

`mlx-lm`'s `generate_step` uses this loop with `prefill_step_size` (default 2048). It
prefills all prompt tokens *except the last* and feeds the last token during the first
decode step, so prefill needs no logits at all. After each chunk, it forces evaluation
of the lazy cache, clears MLX's buffer cache to bound memory, and invokes a progress
callback.

`llama.cpp` splits prompts into `n_batch`-sized calls to `llama_decode` and requests
logits only for the final token. Serving engines add scheduling on top. For example,
vLLM's chunked prefill gives each engine step a token budget, fills it with pending
decode tokens first, and spends the remaining budget on a slice of a waiting prompt.
That keeps inter-token latency bounded for active requests while long prompts prefill.
The `chunked-prefill-scheduling` problem implements that scheduling policy on top of
this single-sequence core.
