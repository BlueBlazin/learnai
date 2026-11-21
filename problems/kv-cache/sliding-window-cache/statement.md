Implement the write operation of a sliding-window KV cache: a fixed buffer that, once
full, overwrites its oldest entries in place, except for the first `keep` tokens,
which stay forever. The buffer is a ring, so its slots are not in sequence order
after it wraps, and the function must also return the cached keys and values in
temporal order.

![This problem is the KV cache, here a fixed ring that overwrites its oldest entries.](assets/map.svg)

## Specification

```python
def rotating_write(cache, k_new, v_new, max_size, keep): ...  # returns (cache, keys, values)
```

A **cache** is a tuple `(k, v, offset)`. `k` and `v` are float32 buffers of shape
`(B, H, max_size, D)`, and `offset` is a Python int: the number of tokens written so
far, over the whole life of the cache. It keeps counting after the buffer is full.
Number the tokens $0, 1, 2, \dots$ in the order they are written.

**What the cache holds.** After $n$ tokens have been written, the cached tokens are

- the **sinks**: tokens $0, \dots, \text{keep} - 1$ (those of them that exist yet), and
- the **window**: the most recent $W = \text{max\_size} - \text{keep}$ of the other
  tokens, that is, every token $t$ with $t \ge \text{keep}$ and $n - W \le t < n$.

Until $n$ exceeds `max_size` that is simply every token.

**Where a token lives.** Token $t$ is stored in slot

$$
\operatorname{slot}(t) = \begin{cases}
t & t < \text{keep}, \\
\text{keep} + (t - \text{keep}) \bmod W & t \ge \text{keep}.
\end{cases}
$$

The sinks occupy slots $0$ to $\text{keep} - 1$. The other slots form a ring of $W$
slots: the first token after the buffer is full goes to slot `keep`, where it
overwrites the oldest token of the window, and so on around. A token's slot never
changes: a write touches only the slots of the tokens it stores, and no entry is ever
moved.

![The buffer for `max_size = 8` and `keep = 2` after 11 tokens. The returned keys are the sinks, then the ring read from the outlined slot.](assets/ring-layout.svg)

**`rotating_write(cache, k_new, v_new, max_size, keep)`** writes the keys and values of
`T` new tokens (numbers `offset` to `offset + T - 1`), each to its slot, and returns a
tuple `(cache, keys, values)`:

- `cache` is `(k, v, offset + T)`. The buffers keep the shape `(B, H, max_size, D)`,
  and slot $\operatorname{slot}(t)$ holds token $t$ for every cached token $t$.
- `keys` and `values` are the entries of the cached tokens in **temporal order**,
  lowest token number first: the sinks, then the window from its oldest token to its
  newest. Their shape is `(B, H, min(offset + T, max_size), D)`. This is what the
  attention of the newest token runs over.

Details that follow from the rules above, stated so that nothing is left to guess:

- `T` may be any size $\ge 1$. If the chunk is longer than the window, its earlier
  tokens are already outside the window when the write completes: they are not
  cached and do not appear in the result (except those that are sinks).
- A chunk may cross the end of the buffer: part of it goes to the last slots and the
  rest continues at slot `keep`.
- `keep = 0` gives a plain ring over the whole buffer.
- Slots that no token has been written to yet hold arbitrary data and are never
  returned.
- `max_size` always equals the buffer's size along axis 2, and `max_size` and `keep`
  are the same in every call on one cache.

**State in, state out.** The function returns the cache to continue with, and the
caller never uses the one it passed in again. A solution may therefore overwrite
slots of the buffers it was given (index assignment in PyTorch and MLX) or return
updated buffers of the same shape (JAX arrays are immutable; an indexed update
returns a new array). Both are correct in every framework, and no state lives
anywhere but in the tuple.

## Shapes

| name | shape | dtype |
|---|---|---|
| `k`, `v` in a cache | `(B, H, max_size, D)` | float32 |
| `offset` in a cache | Python `int` | |
| `k_new`, `v_new` | `(B, H, T, D)` | float32 |
| `keys`, `values` | `(B, H, min(offset + T, max_size), D)` | float32 |

## Example

`max_size = 5` and `keep = 2`, so the ring is slots 2, 3, 4 ($W = 3$). One batch row,
one head, `D = 1`, and the key of token $t$ is the number $t$ itself, so the buffer
can be read as a list of token numbers (the values are $100 + t$ and move in the same
way). The buffer starts as zeros with `offset = 0`.

| write | tokens | key buffer afterwards | `offset` | `keys` returned |
|---|---|---|---|---|
| 1 | 0, 1, 2, 3 | `[0, 1, 2, 3, 0]` | 4 | `[0, 1, 2, 3]` |
| 2 | 4, 5 | `[0, 1, 5, 3, 4]` | 6 | `[0, 1, 3, 4, 5]` |
| 3 | 6, 7 | `[0, 1, 5, 6, 7]` | 8 | `[0, 1, 5, 6, 7]` |
| 4 | 8 | `[0, 1, 8, 6, 7]` | 9 | `[0, 1, 6, 7, 8]` |

- **Write 1** does not fill the buffer. Slot 4 still holds its initial content, and
  only four entries are returned.
- **Write 2** crosses the wrap point. Token 4 goes to slot $2 + (4 - 2) \bmod 3 = 4$,
  the last slot; token 5 goes to slot $2 + (5 - 2) \bmod 3 = 2$, where it overwrites
  token 2, the oldest of the window. The buffer now reads 0, 1, 5, 3, 4 while the
  temporal order is 0, 1, 3, 4, 5.
- **Write 3** puts tokens 6 and 7 in slots 3 and 4. The ring has made a full turn, so
  slot order and temporal order agree again.
- **Write 4** wraps once more. Tokens 0 and 1 have not moved and never will.

## Constraints

- $1 \le B, H \le 4$; $1 \le D \le 16$; $1 \le \text{max\_size} \le 8$;
  $0 \le \text{keep} < \text{max\_size}$; $1 \le T \le 12$; at most 64 tokens are
  written to one cache.
- Entries are copies of the inputs, so everything is compared exactly: after every
  write the tests check `keys` and `values`, the slot of every cached token in the
  returned buffers, the buffer shapes, and that `offset` is a Python int with the
  right value.
- Each write should touch only the slots it stores tokens in. Rebuilding the buffer
  in temporal order on every write is not a ring, and fails the slot check.
- A Python loop over the new tokens is acceptable. No loops over batch rows, heads,
  or the buffer's slots are needed.

## Why this matters

A model has a fixed context length, and a long-running stream (a voice assistant, a
day-long agent session) eventually has more tokens than fit. The cheapest policy is
a sliding window: keep the most recent tokens and drop the rest. A ring buffer makes
eviction free, since the new token simply overwrites the oldest one, where a shifted
buffer would copy the whole window on every step. StreamingLLM (Xiao et al., 2023)
observed that plain windows break down as soon as the first tokens are evicted,
because models dump a large share of their attention on the initial tokens whatever
they say, and that keeping a few of them (four in the paper) as "attention sinks"
restores quality. mlx-lm's `RotatingKVCache(max_size, keep)` is that design, and this
problem's layout is the layout of its buffer. The bugs are quiet ones: evict the
sinks and the model degrades only after the window first wraps, thousands of tokens
into a session; return the ring in slot order and every multi-token step after a
wrap attends through a causal mask that is aligned with the wrong tokens.
