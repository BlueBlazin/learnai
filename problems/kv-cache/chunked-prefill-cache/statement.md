Push a prompt through a transformer language model in consecutive chunks of at most
`chunk_size` tokens, carrying the KV cache from one chunk to the next, and return the
logits that score the first new token together with the cache that covers the whole
prompt. The result must be what a single call on the whole prompt gives; the judge
counts the calls and the tokens.

## Specification

```python
def prefill(lm, prompt, chunk_size): ...
```

`lm` is a [`TinyLM`](../../../python/src/whetstone/kit/tinylm.py), the kit's small
seeded transformer, as in `cached-greedy-decode`:

```python
logits, cache = lm(tokens)           # no cache: `tokens` start the sequence
logits, cache = lm(tokens, cache)    # with a cache: `tokens` continue the cached prefix
```

`logits` is a float32 tensor of shape `(T, vocab_size)` with one row per token passed
in *this* call, and `logits[-1]` scores the token that follows the last one passed.
`cache` covers every token seen so far (`cache.length` of them). A call never modifies
the cache it is given; it returns a new one. A call needs at least one token.

Split the prompt of $P$ tokens into $n = \lceil P / c \rceil$ chunks, where $c$ is
`chunk_size`: chunk $k$ (counting from 0) is `prompt[k * c : (k + 1) * c]`. So every
chunk has exactly $c$ tokens except the last, which has the $P - (n - 1)\,c$ tokens
that remain (between 1 and $c$). Feed the chunks in order, one call per chunk, each
call continuing from the cache the previous call returned.

- `prompt`: a list of ints, $P \ge 1$ token ids.
- `chunk_size`: an int $\ge 1$. It may be larger than $P$; then there is one chunk.
- Returns a tuple `(last_logits, cache)`:
  - `last_logits`: the last row of the logits of the **final** call, a float32 tensor
    of shape `(vocab_size,)`. It scores the first token after the prompt.
  - `cache`: the cache returned by the final call, with `cache.length == P`.

**Cost.** Both counters of the model are fixed by the rule above, and the judge
compares them exactly:

| counter | value | why |
|---|---|---|
| `lm.counters.forward_calls` | $\lceil P / c \rceil$ | one call per chunk, and no call on an empty chunk |
| `lm.counters.tokens_processed` | $P$ | every prompt token goes through the model once |

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| `last_logits` | `(vocab_size,)` | float32 |
| `cache` | a `KVCache` with `length == P` | |

## Example

```python
lm = TinyLM(fw, seed=3)          # vocab_size 64
prompt = [0, 17, 42, 5, 9, 33, 12, 60, 21, 7]

last_logits, cache = prefill(lm, prompt, chunk_size=4)
# last_logits[:4] = [0.9629, 0.8814, 1.8494, 1.0668]     argmax(last_logits) = 18
# cache.length = 10                                       3 calls, 10 tokens processed
```

Ten tokens in chunks of four are $\lceil 10 / 4 \rceil = 3$ chunks:

| call | tokens passed | cache passed in | cache covers afterwards |
|---|---|---|---|
| 1 | `[0, 17, 42, 5]` | none | 4 tokens |
| 2 | `[9, 33, 12, 60]` | 4 tokens | 8 tokens |
| 3 | `[21, 7]` | 8 tokens | 10 tokens |

Call 3 returns two rows of logits. The second one belongs to token 7, the last token
of the prompt, and is `last_logits`: the same 64 numbers as `lm(prompt)[0][-1]`. The
rows returned by calls 1 and 2 score positions inside the prompt and are not needed.

With `chunk_size=5` the same prompt takes exactly two calls of five tokens; a third
call would have nothing to pass. With `chunk_size=16` it is a single call.

## Constraints

- $1 \le P \le 100$ and $1 \le$ `chunk_size` $\le 64$; `lm.max_context` is 128.
- `last_logits` is compared with the logits of a single call on the whole prompt, to a
  relative tolerance of `1e-3` and an absolute tolerance of `1e-4` (splitting the
  calls changes the float32 rounding by about `1e-5`). It must be float32.
- The returned cache must have length $P$, and a token fed after it must get the
  logits it gets after the whole prompt.
- `forward_calls` and `tokens_processed` must equal the values in the table, and no
  call may pass more than `chunk_size` tokens. The chunk layout is checked too: full
  chunks first, the remainder last.

## Why this matters

One call over a long prompt is the fastest way to prefill, and the most expensive in
memory: the activations of every layer, and attention scores of size (new tokens)
$\times$ (all tokens), exist for the whole chunk at once. Engines therefore cap the
chunk. `mlx-lm` prefills in steps of `prefill_step_size` tokens (2048 by default),
evaluating the cache and releasing temporary memory after each step, which is also
where it reports progress on a long prompt; llama.cpp cuts the prompt into batches of
`n_batch` tokens. On a server the same loop, interleaved with the decode steps of
other requests, is *chunked prefill*: a 30,000-token prompt no longer stalls every
stream that is already generating. The loop is short, and each of its bugs is
expensive: a forgotten cache silently changes the model's answer, a re-fed prefix
makes prefill quadratic, and a wrong boundary either crashes on a prompt whose length
happens to be a multiple of the chunk size or defeats the memory bound the chunking
was for.
