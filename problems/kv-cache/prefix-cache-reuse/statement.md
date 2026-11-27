Implement greedy decoding that reuses KV caches from earlier prompts. A store maps previously seen prompts to their KV caches, so a new prompt that starts with the same tokens only needs its remaining suffix processed.

![This problem is Prefix lookup, which finds a stored cache that begins like the prompt.](assets/map.svg)

## Specification

```python
def generate_with_store(lm, store, prompt, max_new_tokens): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py) with the same interface as in `cached-greedy-decode`. Calling `logits, cache = lm(tokens)` starts a new sequence, and `logits, cache = lm(tokens, cache)` continues from an existing cache. Every call requires at least one token, and `logits[-1]` gives the next-token scores. Caches are immutable and support truncation:

```python
shorter = cache.truncate(n)    # returns a new cache covering only the first n tokens, 0 <= n <= cache.length
```

The arguments are:

- `store`: a dict mapping a tuple of token IDs to a `KVCache` covering those exact tokens (`store[key].length == len(key)`). It may be empty.
- `prompt`: a list of $P \ge 1$ token IDs.
- `max_new_tokens`: an integer $g \ge 1$ giving the exact number of tokens to generate. There is no stop token in this problem.

**The reusable prefix.** For any stored key, its *common prefix* with `prompt` is the longest leading sequence of matching tokens. For example, key `(0, 7, 7, 9)` and prompt `[0, 7, 5, 9]` share a common prefix of length 2. Let $m$ be the maximum common prefix length across all keys in `store` ($m = 0$ if `store` is empty). The number of reused tokens is

$$
r = \min(m,\; P - 1).
$$

We cap $r$ at $P - 1$ so at least one prompt token is still fed to `lm`. A stored cache holds keys and values rather than logits, so the model must process the last prompt token to produce logits for the first generated token.

- If $r \ge 1$, pick any stored key whose common prefix with `prompt` has length $m$, truncate its cache to $r$ tokens, and pass `prompt[r:]` with that cache in a single call to `lm`.
- If $r = 0$, pass the entire `prompt` in a single call to `lm` with no cache.

**Generation.** Decode greedily as in `cached-greedy-decode`. At each step, select the token ID with the largest logit in `logits[-1]`, breaking ties in favor of the lower ID. Feed each generated token except the last one back into `lm` with the current cache in its own call. Do not feed the final generated token.

**Returns** a tuple `(tokens, new_store)`:

- `tokens`: the $g$ generated token IDs as a list of Python `int` values, excluding the prompt.
- `new_store`: a new dict containing every entry from `store` (same keys and cache objects), plus `tuple(prompt)` mapped to the cache covering **exactly the prompt** (length $P$, without generated tokens). If `tuple(prompt)` is already in `store`, replace its value in `new_store`. Do not mutate the input `store`.

**Cost limits** per call to `generate_with_store`:

| counter | limit | reason |
|---|---|---|
| `lm.counters.forward_calls` | $g$ | 1 call for the uncached prompt suffix, plus $g - 1$ calls for generated tokens |
| `lm.counters.tokens_processed` | $P - r + g - 1$ | $P - r$ uncached prompt tokens, plus $g - 1$ generated tokens |

## Shapes

| name | shape | dtype |
|---|---|---|
| `store`, `new_store` | dict: `tuple[int, ...]` to `KVCache` | |
| `prompt` | list of length $P$ | Python `int` |
| `logits` (returned by `lm`) | `(T, vocab_size)` for a call of `T` tokens | float32 |
| `tokens` | list of length $g$ | Python `int` |

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

In the first call, `store` is empty, so $r = 0$. All 8 tokens of `a` go into one call, and 3 of the 4 generated tokens are fed back, totaling $8 + 3 = 11$ tokens.

In the second call, `b` shares its first 5 tokens with `tuple(a)` and diverges at index 5 (`21` vs. `33`). That gives $m = 5$ and $r = \min(5, 8) = 5$. The stored cache has length 8, so it is truncated to 5 tokens before feeding `[21, 7, 50, 8]`. This processes $4 + 3 = 7$ tokens instead of $9 + 3 = 12$, and leaves `tuple(a)` unchanged in `store`.

![The second request of the example. The stored cache is cut after the five common tokens, and the remaining four prompt tokens are fed.](assets/prefix-match.svg)

In the third call, `b` matches `tuple(b)` across all 9 tokens, so $m = 9$ and $r = \min(9, 8) = 8$. The cache is truncated to 8 tokens, and the last prompt token (`8`) is fed to obtain next-token logits, totaling $1 + 3 = 4$ tokens.

## Constraints

- $1 \le P$, $1 \le g$, and $P + g \le$ `lm.max_context` (128).
- `store` contains at most a few dozen entries, and every key is non-empty.
- Generated tokens must match uncached greedy decoding exactly. Test models separate the top two logits by more than `1e-3`, so the argmax is independent of how calls are split.
- The new store entry must have length $P$, and feeding a follow-up token after it must match the logits obtained after the full prompt within a relative tolerance of `1e-3`.
- Both cost counters are checked per call. Exceeding either limit fails even if the generated tokens are correct.

## Why this matters

Production LLM servers repeatedly see prompts with shared prefixes, such as a common system prompt, tool definitions, few-shot examples, or an ongoing conversation with one new turn. Because prefill cost scales with prompt length, reusing a cached prefix can cut time-to-first-token on long contexts from seconds to milliseconds.

Inference engines including `llama.cpp` (`cache_prompt`), `mlx-lm`, vLLM (automatic prefix caching), and hosted prompt-caching APIs all rely on this mechanism. Small edge-case bugs have outsized effects. Matching only exact prompts throws away almost all cache hits, forgetting to truncate a diverging cache makes the model condition on tokens that are not in the prompt, and reusing all $P$ tokens of an exact match leaves no logits to sample the first token from.
