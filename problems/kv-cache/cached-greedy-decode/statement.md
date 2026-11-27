Implement greedy decoding for a transformer language model using its KV cache. Pass the prompt in a single call, and then pass one generated token per step so every token goes through the model at most once. Your function must produce the exact same tokens as uncached greedy decoding while staying within strict call and token limits.

![This problem is the whole loop, from the prompt to the last generated token.](assets/map.svg)

## Specification

```python
def greedy_decode(lm, prompt, max_new_tokens, eos_id): ...
```

`lm` is a [`TinyLM`](../../../python/src/learnai/kit/tinylm.py), a small seeded decoder-only transformer from the kit. You can call it in two ways:

```python
logits, cache = lm(tokens)           # no cache: `tokens` is the whole sequence so far
logits, cache = lm(tokens, cache)    # with a cache: `tokens` continue the cached prefix
```

- `tokens` is a list of ints (or a 1-D integer tensor) containing `T` tokens, with $T \ge 1$.
- `logits` has one row per token passed in that call. Row `i` scores the token that follows `tokens[i]`, so `logits[-1]` scores the next token of the sequence.
- `cache` stores the attention keys and values for every token the model has seen so far (`cache.length` tokens). The model never mutates a cache in place and returns a new cache on every call.
- `lm.counters.forward_calls` counts how many times `lm` was called. `lm.counters.tokens_processed` sums the number of tokens passed across all calls.

At each step of greedy decoding, the next token is the argmax of the next-token logits given the prompt and all tokens generated so far. Implement `greedy_decode` with these arguments and behavior:

- `prompt`: the initial sequence of $P \ge 1$ token ids.
- `max_new_tokens`: the maximum number of tokens to generate, with `max_new_tokens` $\ge 1$.
- `eos_id`: an end-of-sequence token id (`int`), or `None`. If a generated token equals `eos_id`, stop immediately and include that token as the last element of the returned list. An `eos_id` inside `prompt` does not stop generation. If `eos_id` is `None`, generate exactly `max_new_tokens` tokens.
- Return a list of $g$ generated token ids (excluding the prompt), where $1 \le g \le$ `max_new_tokens`.

**Cost limits.** For a call that returns $g$ tokens from a prompt of length $P$, the counters must not exceed these exact limits:

| counter | limit | why |
|---|---|---|
| `forward_calls` | $g$ | one call for the prompt (which yields the first token), then one call per additional token |
| `tokens_processed` | $P + g - 1$ | each prompt token once, and each generated token once except the last |

Do not feed the last generated token back into the model because nothing uses its logits. This applies both when hitting `eos_id` and when reaching `max_new_tokens`. You must also pass the entire prompt in one call rather than token by token. Both limits are the theoretical minimum, so they are exact.

## Shapes

| name | shape | dtype |
|---|---|---|
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| result | list of $g$ ints | Python `int` |

## Example

```python
lm = TinyLM(fw, seed=3)          # vocab_size 64

greedy_decode(lm, [0, 17, 42, 5], 8, eos_id=1)
# [42, 62, 33, 33, 40, 32, 45, 37]        8 calls, 11 tokens processed

greedy_decode(lm, [0, 17, 42, 5], 8, eos_id=40)
# [42, 62, 33, 33, 40]                    5 calls, 8 tokens processed
```

The first call runs as follows:

| call | tokens passed | cache covers afterwards | argmax of `logits[-1]` |
|---|---|---|---|
| 1 | `[0, 17, 42, 5]` | 4 tokens | 42 |
| 2 | `[42]` | 5 tokens | 62 |
| 3 | `[62]` | 6 tokens | 33 |
| 4 | `[33]` | 7 tokens | 33 |
| 5 | `[33]` | 8 tokens | 40 |
| 6 | `[40]` | 9 tokens | 32 |
| 7 | `[32]` | 10 tokens | 45 |
| 8 | `[45]` | 11 tokens | 37 |

After 8 calls, 8 tokens have been generated and the function returns. Token 37 is never passed to the model, so the total is $4 + 7 = 11$ processed tokens. Without a cache, call $k$ would pass all $3 + k$ tokens, totaling 60 tokens.

![The first four calls of the example, without and with a cache. A filled cell is a token the cache already covers.](assets/tokens-per-call.svg)

In the second example, call 5 generates token 40, which equals `eos_id`. Generation stops immediately and returns 5 tokens after 5 calls and $4 + 4 = 8$ processed tokens.

## Constraints

- $1 \le P$, $1 \le$ `max_new_tokens`, and $P +$ `max_new_tokens` $\le$ `lm.max_context` (128).
- Generated tokens are compared exactly against uncached greedy decoding. In the test models, the top two logits differ by more than `1e-3`, so the argmax does not depend on how calls are split.
- Both counters are checked against the exact limits above. Exceeding either limit fails even if the returned tokens are right.
- If two logits tie, the argmax is the lower token id (ties do not occur in the tests).

## Why this matters

Every LLM inference engine uses this two-phase loop. A *prefill* pass processes the entire prompt at once, and subsequent *decode* steps process one token at a time against the cached keys and values. Without a KV cache, step $k$ recomputes attention over all previous tokens, turning $n$ generated tokens into $O(n^2)$ token passes instead of $O(n)$.

Off-by-one bugs around this loop are common in production servers. Feeding the wrong token after prefill shifts every prediction by one position. Processing the prompt one token at a time throws away the parallelism that keeps time-to-first-token low. Running an extra forward pass on the final token wastes latency on every short response.
