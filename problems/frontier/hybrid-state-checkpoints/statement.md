Reuse earlier work for a model whose memory of a prefix is a fixed-size recurrent
state instead of a KV cache. A KV cache can be cut back to any shorter prefix; a
state cannot give tokens back. The only prefixes you can return to are the ones whose
state you saved, so prefix reuse becomes a question of where to take checkpoints,
which one to resume from, and which to keep when the store is full.

## Specification

```python
def generate_with_checkpoints(lm, store, prompt, max_new_tokens, interval, max_checkpoints): ...
```

`lm` is a [`RecurrentLM`](../../../python/src/whetstone/kit/recurrent.py). It is
called like the `TinyLM` of `prefix-cache-reuse`:

```python
logits, state = lm(tokens)  # tokens are the whole sequence so far
logits, state = lm(tokens, state)  # tokens continue the sequence that `state` has read
```

A call needs at least one token, `logits[-1]` scores the next token, and the state
passed in is never modified. A `RecurrentState` has the same size after 4 tokens as
after 400 (`state.length` is the number of tokens it has read). It has no working
`truncate`: the tokens are folded into it. A call returns the state after its
**last** token only, so the state at a position is available only if a call ends
there. Continuing from a saved state gives exactly the same bits as reading the whole
sequence in one call.

- `store`: a dict that maps a tuple of token ids to the state after exactly those
  tokens (`store[key].length == len(key)`). Its **order is its recency**: Python
  dicts keep insertion order, the first key is the least recently used checkpoint and
  the last key the most recently used. It may be empty.
- `prompt`: a list of $P \ge 1$ token ids.
- `max_new_tokens`: an int $g \ge 1$; exactly $g$ tokens are generated (no stop
  token).
- `interval`: an int $n \ge 1$, the spacing of checkpoints.
- `max_checkpoints`: an int $\ge 1$, the most entries the returned store may hold.

**The checkpoint to resume from.** A stored key is *usable* when it is a proper
prefix of the prompt: `len(key) < P` and `key == tuple(prompt[:len(key)])`. Resume
from the longest usable key; let $r$ be its length, or $r = 0$ when no key is usable
(start from nothing). A key that shares its first tokens with the prompt and then
continues differently is not usable, however many tokens it shares, and neither is
the key of the whole prompt: one prompt token must be left to feed, because logits
come only out of a model call.

**Checkpoints to save.** Let the *sequence* be the prompt followed by the generated
tokens. Save the state at every sequence length $c$ that the model passes through
after resuming and that is a multiple of `interval`, and at the end of the prompt:

$$
c \in \{\, c : r < c \le P + g - 1,\ c \bmod n = 0 \,\} \;\cup\; \{P\},
$$

each under the key `tuple(sequence[:c])`. (The last generated token is never fed, so
the model reaches length $P + g - 1$ at most.) Multiples of `interval` are counted
from the start of the sequence, not from $r$.

**Calls.** The prompt tokens after the resumed checkpoint are fed in segments that
end at exactly those lengths $c \le P$, one call per segment, in order. Generation is
greedy, as in `prefix-cache-reuse`: each new token is the argmax of the next-token
logits (the lower id on a tie), and each generated token except the last is fed in a
call of its own.

**The returned store** is a new dict (do not modify `store`), built by these rules:

1. It starts as `store`, in the same order.
2. The checkpoint that is resumed from becomes the most recent entry (it moves to
   the end).
3. Each checkpoint is saved in the order the model reaches it. Saving puts the key at
   the end; a key that is already present is replaced and moved to the end.
4. Finally, while the store holds more than `max_checkpoints` entries, the first
   (least recently used) entry is removed.

Entries of `store` that are neither resumed from, saved again, nor evicted keep
their place and their state object.

Returns the tuple `(tokens, new_store)`; `tokens` is the list of the $g$ generated
tokens as Python ints.

**Cost limits** for one call of `generate_with_checkpoints`:

| counter | limit | why |
|---|---|---|
| `lm.counters.tokens_processed` | $P - r + g - 1$ | the prompt tokens after the resumed checkpoint, and each generated token except the last |
| `lm.counters.forward_calls` | $s + g - 1$ | $s$ is the number of prompt segments: the multiples of `interval` in $(r, P)$, plus one for the segment that ends at $P$ |
| entries of `new_store` | `max_checkpoints` | the budget |

## Shapes

| name | shape | dtype |
|---|---|---|
| `store`, `new_store` | dict: tuple of ints to `RecurrentState` | |
| `prompt` | list of $P$ ints | Python `int` |
| `logits` (from `lm`) | `(T, vocab_size)` for a call with `T` tokens | float32 |
| `tokens` | list of $g$ ints | Python `int` |

## Example

`interval = 4`, `max_checkpoints = 4`, on `RecurrentLM(seed=3)`:

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

**First request.** Nothing is stored, $r = 0$. The prompt is fed as `a[0:4]`,
`a[4:8]`, `a[8:10]`, and the states after 4, 8, and 10 tokens are saved. Three of
the four generated tokens are fed, which takes the sequence to lengths 11, 12, 13;
length 12 is a multiple of 4, so the state after `a + [26, 6]` is saved too. That is
$3 + 3 = 6$ calls and $10 + 3 = 13$ tokens.

**Second request.** `b` agrees with `a` on six tokens. The stored keys of 8, 10, and
12 tokens all contain `a[6]`, which `b` does not, so none is a prefix of `b`; the
key of 4 tokens is. $r = 4$, and that entry moves to the end. The prompt is fed as
`b[4:8]` and `b[8:9]`, saving states under `b[:8]` and `b`. Two generated tokens
are fed (lengths 10 and 11, no multiple of 4). That is $2 + 2 = 4$ calls and
$5 + 2 = 7$ tokens instead of $9 + 2 = 11$.

The store now has six entries, in order: `a[:8]`, `a`, `a + [26, 6]`, `a[:4]`,
`b[:8]`, `b`. The budget is 4, so the first two are removed.

Note what a KV cache would have done: reuse 6 tokens by truncating. Here the two
shared tokens after the checkpoint are computed again, and that is the price of the
interval.

## Constraints

- $1 \le P$, $1 \le g$, $P + g \le$ `lm.max_context` (256); $1 \le$ `interval`;
  $1 \le$ `max_checkpoints`. The store has at most a few dozen entries, and its keys
  are non-empty.
- The generated tokens are compared exactly with greedy decoding of the whole
  sequence without any saved state.
- The keys of the returned store are compared exactly, **in order**. Every state in
  it is compared bit for bit with the state of a fresh run over its key; states the
  model returned for the right prefix pass.
- The three limits above are checked per call, after the tokens and the store.
  `tokens_processed` cannot be beaten with the checkpoints the store holds.

## Why this matters

State-space and linear-attention layers keep per-request memory constant and make
prefill linear in the prompt length, and hybrid models that interleave them with
attention layers are now served by the same engines as transformers. Prefix caching
is the feature that breaks. An engine that treats the recurrent state like a KV
cache and "truncates" to the shared prefix continues from a state that has read
tokens the new prompt does not contain, and the model answers a prompt nobody sent,
fluently. An engine that notices the problem and gives up re-reads the whole
conversation on every turn, which shows up as a time to first token that grows with
the chat. Checkpoints are the middle path, and they are a budget: each one is a full
copy of the state, far larger than one token's keys and values, so how many to keep
and which to drop is the tuning knob that such engines expose.
