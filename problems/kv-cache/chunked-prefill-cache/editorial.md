## The idea

A causal transformer computes position $i$ from tokens $0..i$ only, and the KV cache
holds everything later positions need from earlier ones. So it makes no difference to
the result whether ten tokens arrive in one call or as 4 + 4 + 2: as long as each call
continues from the cache of the one before, every token sees exactly the same context.
What does change is the size of the work done at once. A call over $T$ new tokens with
$L$ cached ones builds activations for $T$ positions and attention scores of shape
$T \times (L + T)$; capping $T$ at `chunk_size` caps that peak, while the total work
stays the same.

The loop keeps one invariant: *before each call, the cache covers exactly the tokens
before the chunk*. Then the chunk is all that needs to go in, each token goes in once,
and after the last chunk the cache covers the prompt. The logits of a call have one
row per token of that call, so the row that scores the token after the prompt is the
last row of the last call. Every earlier row predicts a token the prompt already
contains.

## Reference solution (PyTorch)

```python
def prefill(lm, prompt, chunk_size):
    cache = None
    for start in range(0, len(prompt), chunk_size):
        logits, cache = lm(prompt[start : start + chunk_size], cache)
    return logits[-1], cache
```

`range(0, P, c)` has $\lceil P / c \rceil$ elements and never starts at or beyond
$P$, so no chunk is empty; the slice clips the last chunk to the remainder. The JAX
and MLX versions are identical, because the function only moves tokens and caches
around.

## Common mistakes

- **Re-feeding the prefix** (`lm(prompt[:end])` for a growing `end`). The logits are
  right and nothing looks wrong, but the model processes $c + 2c + 3c + \dots$ tokens:
  436 instead of 100 for a 100-token prompt in chunks of 16, and each call is as large
  as the prefix, so the memory cap is gone as well.
- **Dropping the cache** (`lm(chunk)` without it). Every chunk is then run as if it
  were the beginning of a sequence. The output has the right shape and plausible
  values; it is the answer to a different, shorter prompt.
- **A trailing empty call.** A loop bound of `len(prompt) + 1`, or a `while` loop that
  tests `start <= len(prompt)`, works until the prompt length is a multiple of
  `chunk_size`; then the last slice is empty, and the model rejects an empty call.
  This is the bug that passes every test written with "a prompt of 10 and a chunk of
  4".
- **Returning the wrong row.** Keeping the first chunk's logits (or the whole
  `(T, vocab_size)` block of the last chunk) instead of the last row of the last
  chunk. With a single chunk the first mistake is invisible.
- **Rounding the number of chunks down** and letting the last call absorb the
  remainder. The numbers are right, but the last call has up to $2c - 1$ tokens, which
  breaks the one promise `chunk_size` makes. Splitting from the end, with the short
  chunk first, respects the cap but not the specified layout.
- **Ignoring `chunk_size`** and feeding one token per call. Correct, and as slow as
  prefill can be: $P$ sequential calls where $\lceil P / c \rceil$ parallel ones
  suffice.

## Complexity

$\lceil P / c \rceil$ forward calls and $P$ tokens processed. The attention work
inside the model is the same $O(P^2)$ as for a single call; chunking only bounds how
much of it is in flight at once, to $O(c \cdot P)$ scores per layer instead of
$O(P^2)$.

## In the wild

`mlx-lm`'s `generate_step` has this loop with `prefill_step_size` (default 2048). It
differs in one detail: it prefills all prompt tokens *but the last*, then feeds the
last token in the first decode step, so its prefill needs no logits at all. After
each chunk it forces the lazily computed cache to be evaluated and clears MLX's
buffer cache, which is what actually bounds the memory, and calls a progress
callback. llama.cpp splits a prompt into `n_batch`-sized calls of `llama_decode` and
asks for logits only for the final token. Serving engines add scheduling: vLLM's
chunked prefill gives each engine step a token budget and fills it with pending
decode tokens first and a slice of a waiting prompt second, so the time between
tokens of running requests stays bounded while a long prompt is being read. The
`chunked-prefill-scheduling` problem builds that policy; this one is its
single-sequence core.
