Implement prefix reuse for a model whose memory of past tokens is a fixed-size recurrent state rather than a KV cache. Because a recurrent state cannot be truncated to an earlier prefix, you must decide where to save checkpoints, which checkpoint to resume from, and which checkpoints to keep when the store is full.

## Specification

```python
def generate_with_checkpoints(lm, store, prompt, max_new_tokens, interval, max_checkpoints): ...
```

`lm` is a [`RecurrentLM`](../../../python/src/learnai/kit/recurrent.py) model. You call it the same way as `TinyLM` in `prefix-cache-reuse`:

```python
logits, state = lm(tokens)  # tokens are the whole sequence so far
logits, state = lm(tokens, state)  # tokens continue the sequence that `state` has read
```

Each call requires at least one token, `logits[-1]` scores the next token, and the input `state` is never modified. A `RecurrentState` has the same byte size after 4 tokens as after 400, and `state.length` records how many tokens it has read. Tokens are folded into the state with no way to truncate it back. A call only returns the state after its **last** token, so a state at a given position is available only if a call ends there. Resuming from a saved state produces the exact same output as reading the full sequence in a single call.

Inputs:

- `store`: a dict mapping a tuple of token IDs to the `RecurrentState` after reading those exact tokens (`store[key].length == len(key)`). Python dicts preserve insertion order, which tracks recency here. The first key is the least recently used checkpoint, and the last key is the most recently used. `store` may be empty.
- `prompt`: a list of $P \ge 1$ token IDs.
- `max_new_tokens`: an integer $g \ge 1$. Generate exactly $g$ tokens, with no stop token.
- `interval`: an integer $n \ge 1$ giving the token spacing between checkpoints.
- `max_checkpoints`: an integer $\ge 1$ giving the maximum number of entries allowed in the returned store.

**Choosing a checkpoint to resume from.** A stored key is *usable* if it is a proper prefix of `prompt`, meaning `len(key) < P` and `key == tuple(prompt[:len(key)])`. Resume from the longest usable key, and let $r$ be its length. If no key is usable, start from `None` with $r = 0$. A key that shares its opening tokens with `prompt` and then diverges is not usable, regardless of how many tokens it shares. The key equal to the entire `prompt` is also not usable because at least one prompt token must remain to feed into `lm` to obtain next-token logits.

**Choosing checkpoints to save.** Let the *sequence* be `prompt` followed by the generated tokens. Save the state at every sequence length $c$ that the model reaches after resuming that is either a multiple of `interval` or the end of the prompt:

$$
c \in \{\, c : r < c \le P + g - 1,\ c \bmod n = 0 \,\} \;\cup\; \{P\}.
$$

Store each checkpoint under the key `tuple(sequence[:c])`. Because the final generated token is never fed back into the model, the model reaches at most length $P + g - 1$. Multiples of `interval` are counted from the start of the sequence, not from $r$.

**Model calls.** Feed the remaining prompt tokens after the resumed checkpoint in segments ending at each target length $c \le P$, using one call per segment in order. Generation is greedy, matching `prefix-cache-reuse`. Each new token is the argmax of the next-token logits (choosing the lower ID on a tie), and each generated token except the last is fed in its own single-token call.

**Building the returned store.** Return a new dict without modifying `store`, constructed in four steps:

1. Copy `store` in its existing order.
2. If you resumed from a checkpoint, move its key to the end so it becomes the most recently used entry.
3. Save each new checkpoint in the order the model reaches it. Saving places the key at the end. If the key is already present, replace its value and move it to the end.
4. While the store has more than `max_checkpoints` entries, remove the first (least recently used) entry.

Entries in `store` that are not resumed, re-saved, or evicted keep their relative order and original state objects.

Return the tuple `(tokens, new_store)`, where `tokens` is the list of $g$ generated tokens as Python `int` values.

**Cost limits** per call to `generate_with_checkpoints`:

| counter | limit | why |
|---|---|---|
| `lm.counters.tokens_processed` | $P - r + g - 1$ | prompt tokens after the resumed checkpoint, plus every generated token except the last |
| `lm.counters.forward_calls` | $s + g - 1$ | $s$ is the number of prompt segments (multiples of `interval` in $(r, P)$, plus one for the segment ending at $P$) |
| entries in `new_store` | `max_checkpoints` | store capacity |

## Shapes

| name | shape | dtype |
|---|---|---|
| `store`, `new_store` | dict mapping tuple of ints to `RecurrentState` | |
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| `tokens` | list of $g$ ints | Python `int` |

## Example

Using `interval = 4`, `max_checkpoints = 4`, and `RecurrentLM(seed=3)`:

```python
a = [0, 17, 42, 5, 9, 33, 12, 60, 21, 7]  # P = 10
b = [0, 17, 42, 5, 9, 33, 50, 8, 3]  # P = 9, shares 6 tokens with a

tokens, store = generate_with_checkpoints(lm, {}, a, 4, 4, 4)
# tokens = [26, 6, 7, 26]              6 calls, 13 tokens processed
# key lengths in store: [4, 8, 10, 12]

tokens, store = generate_with_checkpoints(lm, store, b, 3, 4, 4)
# tokens = [11, 32, 38]                4 calls, 7 tokens processed
# key lengths in store: [12, 4, 8, 9]
```

**First request.** The store is empty, so $r = 0$. The prompt is fed in three segments: `a[0:4]`, `a[4:8]`, and `a[8:10]`. States are saved at lengths 4, 8, and 10. Three of the four generated tokens are fed back in single-token calls, advancing the sequence to lengths 11, 12, and 13. Because 12 is a multiple of 4, the state for `a + [26, 6]` is also saved. In total, this takes $3 + 3 = 6$ calls and $10 + 3 = 13$ tokens.

**Second request.** Prompt `b` matches `a` for 6 tokens. The stored keys of lengths 8, 10, and 12 all include `a[6]`, which differs from `b[6]`, so none of them is a prefix of `b`. Only the 4-token key `a[:4]` is usable. Resuming from it gives $r = 4$ and moves `a[:4]` to the end of the store. The remaining prompt is fed as `b[4:8]` and `b[8:9]`, saving states under `b[:8]` and `b`. Two generated tokens are fed at lengths 10 and 11, neither of which is a multiple of 4. This takes $2 + 2 = 4$ calls and $5 + 2 = 7$ tokens instead of $9 + 2 = 11$.

![The prompt shares six tokens with the stored sequence. Only the state after 4 tokens is a prefix of it, so that one is resumed.](assets/checkpoint-resume.svg)

Before eviction, the store holds six entries in order: `a[:8]`, `a`, `a + [26, 6]`, `a[:4]`, `b[:8]`, and `b`. Because `max_checkpoints = 4`, the first two entries are removed.

A KV cache would have reused all 6 shared tokens by truncating. With checkpoints every 4 tokens, the 2 shared tokens after length 4 must be recomputed.

## Constraints

- $1 \le P$, $1 \le g$, and $P + g \le$ `lm.max_context` (256).
- $1 \le$ `interval` and $1 \le$ `max_checkpoints`.
- `store` has at most a few dozen entries, and all its keys are non-empty.
- Generated tokens are compared exactly against greedy decoding of the full sequence with no saved state.
- Returned store keys are compared exactly and **in order**. Each stored state is compared bit for bit against a fresh run over its key.
- All three cost limits are checked per call after verifying tokens and store contents. `tokens_processed` cannot be improved beyond the limit given the checkpoints in `store`.

## Why this matters

State-space and linear-attention layers keep per-request memory constant and run prefill in linear time. Hybrid models that interleave these layers with standard attention are now served by the same inference engines as transformers, and prefix caching is where the two diverge.

An engine that treats a recurrent state like a KV cache and tries to reuse a longer shared prefix will continue from a state that already absorbed mismatched tokens, causing the model to answer a different prompt fluently. An engine that disables prefix caching for recurrent layers instead re-reads the entire conversation on every turn, making time to first token grow with chat length. Checkpointing is the middle ground. Because each checkpoint copies an entire state—far larger than a single token's keys and values—deciding how often to save and which checkpoints to evict is a core memory-compute tradeoff exposed by hybrid serving engines.
