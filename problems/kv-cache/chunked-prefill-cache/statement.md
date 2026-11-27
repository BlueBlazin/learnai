Run a prompt through a transformer language model in consecutive chunks of at most
`chunk_size` tokens, carrying the KV cache from each chunk to the next. Return the
logits that score the first generated token along with the final cache covering the
entire prompt. The output must match a single call over the full prompt, and the judge
verifies both the call count and the total tokens processed.

![This problem is Prefill, here done in chunks that share one cache.](assets/map.svg)

## Specification

```python
def prefill(lm, prompt, chunk_size): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py), the kit's small
seeded transformer from `cached-greedy-decode`. You call it with a list of token IDs
and an optional cache:

```python
logits, cache = lm(tokens)           # no cache: `tokens` start the sequence
logits, cache = lm(tokens, cache)    # with a cache: `tokens` continue the cached prefix
```

When called with `T` tokens, `lm` returns `logits` of shape `(T, vocab_size)` and a
new `cache`. Each row of `logits` scores the token that follows the corresponding
input token, so `logits[-1]` scores the token after the last input token. The returned
`cache` covers every token seen so far (`cache.length` tokens in total). Calls never
modify the cache passed in, and every call requires at least one token.

Let $P$ be the number of tokens in `prompt` and $c$ be `chunk_size`. Split `prompt`
into $n = \lceil P / c \rceil$ consecutive chunks, where chunk $k$ (0-indexed) is
`prompt[k * c : (k + 1) * c]`. Every chunk contains $c$ tokens except possibly the
last, which contains the remaining $P - (n - 1)\,c$ tokens (between $1$ and $c$).
Feed the chunks to `lm` in order, passing the cache returned by each call into the
next call.

- `prompt`: a list of $P \ge 1$ token IDs.
- `chunk_size`: an integer $\ge 1$. If `chunk_size` is larger than $P$, the entire
  prompt runs in one chunk.
- Returns `(last_logits, cache)`:
  - `last_logits`: the last row of the **final** call's `logits`, which scores the
    first token after the prompt.
  - `cache`: the cache returned by the final call, with `cache.length == P`.

**Cost.** The chunking rule fixes both model counters, and the judge checks them for
exact equality.

| counter | value | why |
|---|---|---|
| `lm.counters.forward_calls` | $\lceil P / c \rceil$ | one call per chunk, with no empty calls |
| `lm.counters.tokens_processed` | $P$ | every prompt token passes through the model once |

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| `last_logits` | `(vocab_size,)` | float32 |
| `cache` | `KVCache` with `length == P` | |

## Example

```python
lm = TinyLM(fw, seed=3)          # vocab_size 64
prompt = [0, 17, 42, 5, 9, 33, 12, 60, 21, 7]

last_logits, cache = prefill(lm, prompt, chunk_size=4)
# last_logits[:4] = [0.9629, 0.8814, 1.8494, 1.0668]     argmax(last_logits) = 18
# cache.length = 10                                       3 calls, 10 tokens processed
```

A 10-token prompt with `chunk_size=4` splits into $\lceil 10 / 4 \rceil = 3$ chunks.

| call | tokens passed | cache passed in | cache covers afterwards |
|---|---|---|---|
| 1 | `[0, 17, 42, 5]` | none | 4 tokens |
| 2 | `[9, 33, 12, 60]` | 4 tokens | 8 tokens |
| 3 | `[21, 7]` | 8 tokens | 10 tokens |

Call 3 returns two rows of logits. Its second row corresponds to token 7 (the last
token of the prompt) and becomes `last_logits`, matching the 64 values of
`lm(prompt)[0][-1]`. The logits from calls 1 and 2 score positions inside the prompt
and are discarded.

With `chunk_size=5`, the same prompt takes two calls of 5 tokens, with no empty third
call. With `chunk_size=16`, it runs in a single call.

## Constraints

- $1 \le P \le 100$ and $1 \le$ `chunk_size` $\le 64$. `lm.max_context` is 128.
- `last_logits` must be float32 and match the logits of a single full-prompt call
  within relative tolerance `1e-3` and absolute tolerance `1e-4`. Splitting the calls
  changes float32 rounding by about `1e-5`.
- The returned `cache` must have `length == P`. Passing another token to `lm` with
  this cache must yield the same logits as passing it after a single-call prefill.
- `forward_calls` and `tokens_processed` must equal the values in the cost table. No
  call may pass more than `chunk_size` tokens, and full chunks must come first with
  any remainder last.

## Why this matters

Prefilling a long prompt in one call is fast, but it creates a large memory spike.
Activations for every layer and attention score matrices of size
$(\text{new tokens}) \times (\text{all tokens})$ must fit in memory at once.
Inference engines cap the chunk size to bound this peak. For example, `mlx-lm`
prefills in steps of `prefill_step_size` tokens (2048 by default), evaluating the
cache, freeing temporary buffers, and reporting progress after each step. Similarly,
`llama.cpp` splits prompts into batches of `n_batch` tokens.

Serving engines extend this same loop into *chunked prefill*, interleaving prompt
chunks with decode steps from other requests so a 30,000-token prompt does not stall
active streams. Even though the loop is short, every mistake is costly. Dropping the
cache silently corrupts the model's answer, re-feeding the prefix makes prefill
quadratic, and an off-by-one boundary either crashes when $P$ is a multiple of
`chunk_size` or breaks the memory bound.
