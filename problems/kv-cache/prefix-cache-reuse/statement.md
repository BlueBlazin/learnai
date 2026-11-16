Generate greedily from a prompt, but start from work that is already done: a store
maps prompts seen earlier to their KV caches, and a new prompt that begins with the
same tokens as a stored one only needs its remaining tokens processed. Find the
longest reusable prefix, cut the stored cache down to it, prefill the rest, generate,
and add the new prompt's cache to the store.

## Specification

```python
def generate_with_store(lm, store, prompt, max_new_tokens): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py), used as in
`cached-greedy-decode`: `logits, cache = lm(tokens)` starts a sequence,
`logits, cache = lm(tokens, cache)` continues the cached prefix, a call needs at
least one token, and `logits[-1]` scores the next token. Caches are immutable, and
one more operation matters here:

```python
shorter = cache.truncate(n)    # a new cache covering only the first n tokens, 0 <= n <= cache.length
```

- `store`: a dict that maps a tuple of token ids to a `KVCache` covering exactly
  those tokens (`store[key].length == len(key)`). It may be empty.
- `prompt`: a list of ints, $P \ge 1$ token ids.
- `max_new_tokens`: an int $g \ge 1$. Exactly $g$ tokens are generated (there is no
  stop token in this problem).

**The reusable prefix.** For a stored key, the *common prefix* with the prompt is the
longest run of leading tokens that are equal in both: for key `(0, 7, 7, 9)` and
prompt `[0, 7, 5, 9]` it has length 2. Let $m$ be the largest common prefix length
over all keys of the store (0 for an empty store). The number of reused tokens is

$$
r = \min(m,\; P - 1).
$$

The cap at $P - 1$ keeps one prompt token to feed: logits come only out of a model
call, and the first generated token needs the logits of the last prompt token.

- If $r \ge 1$: take an entry whose common prefix with the prompt is $m$ (if several
  have it, any of them), truncate its cache to $r$ tokens, and continue from it with
  `prompt[r:]` in **one** call.
- If $r = 0$: process the whole prompt in one call, without a cache.

**Generation** is greedy, as in `cached-greedy-decode`: each new token is the argmax
of the next-token logits (the lower id on a tie), each generated token except the
last is fed with the cache in a call of its own, and the last one is never fed.

**Returns** a tuple `(tokens, new_store)`:

- `tokens`: the $g$ generated tokens, a list of Python ints (without the prompt).
- `new_store`: a dict with every entry of `store`, the same cache objects under the
  same keys, plus the key `tuple(prompt)` mapped to a cache that covers **exactly the
  prompt** (length $P$, without the generated tokens). If `tuple(prompt)` was already
  a key, its cache is replaced. Do not modify the `store` argument; build a new dict.

**Cost limits** for one call of `generate_with_store`:

| counter | limit | why |
|---|---|---|
| `lm.counters.forward_calls` | $g$ | one call for the rest of the prompt, then one per generated token except the last |
| `lm.counters.tokens_processed` | $P - r + g - 1$ | the $P - r$ prompt tokens the store does not cover, and each generated token except the last |

## Shapes

| name | shape | dtype |
|---|---|---|
| `store`, `new_store` | dict: tuple of ints to `KVCache` | |
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| `tokens` | list of $g$ ints | Python `int` |

## Example

```python
lm = TinyLM(fw, seed=3)
a = [0, 17, 42, 5, 9, 33, 12, 60]
b = [0, 17, 42, 5, 9, 21, 7, 50, 8]

tokens, store = generate_with_store(lm, {}, a, 4)
# tokens = [4, 61, 60, 26]          4 calls, 11 tokens processed
# store  = {(0, 17, 42, 5, 9, 33, 12, 60): <cache of length 8>}

tokens, store = generate_with_store(lm, store, b, 4)
# tokens = [35, 45, 60, 60]         4 calls, 7 tokens processed
# store has two keys: tuple(a) with length 8, tuple(b) with length 9

tokens, store = generate_with_store(lm, store, b, 4)
# tokens = [35, 45, 60, 60]         4 calls, 4 tokens processed
```

The first request finds nothing to reuse: $r = 0$, the eight prompt tokens go in as
one call, and three of the four generated tokens are fed, $8 + 3 = 11$ tokens.

The second prompt shares its first five tokens with the stored key and then differs
(`21` against `33`), so $m = 5$ and $r = \min(5, 8) = 5$. The stored cache covers
eight tokens, three of which do not belong to this prompt: it is truncated to 5, and
the call passes `[21, 7, 50, 8]`. That is $4 + 3 = 7$ tokens instead of
$9 + 3 = 12$. The stored entry for `a` is untouched.

The third request repeats the second prompt. Now a key matches all nine tokens,
$m = 9$, and $r = \min(9, 8) = 8$: the entry is truncated to 8 tokens and the last
prompt token, `8`, is fed again to obtain logits. $1 + 3 = 4$ tokens.

## Constraints

- $1 \le P$, $1 \le g$, and $P + g \le$ `lm.max_context` (128). The store has at most
  a few dozen entries, and its keys are non-empty.
- The generated tokens are compared exactly with greedy decoding that uses no cache
  at all. The test models keep the two largest logits more than `1e-3` apart, so the
  argmax does not depend on how the calls are split.
- The new entry must have length $P$, and a token fed after it must get the logits it
  gets after the whole prompt (to a relative tolerance of `1e-3`).
- Both counters are compared with the limits above, per call. A solution over a
  limit fails even when its tokens are right. The limits cannot be beaten: the store
  holds caches, not logits.

## Why this matters

Most prompts a server sees begin like one it has seen before: the same system prompt
and tool definitions for every user, the same few-shot examples, the same
conversation plus one more turn, the same file with one edited line. Prefill cost is
proportional to prompt length, so reusing the cached prefix is the difference between
a time to first token of seconds and of milliseconds on a long context. It is what
llama.cpp's `cache_prompt` does within a slot, what `mlx-lm`'s server does with its
prompt cache, what vLLM's automatic prefix caching does across requests, and what
hosted APIs sell as prompt caching. The three rules of this problem are where
implementations go wrong: matching only identical prompts wastes nearly all of the
benefit, forgetting to truncate a cache that continues differently makes the model
answer a prompt nobody sent, and reusing *everything* of an identical prompt leaves
nothing to compute the first token from.
