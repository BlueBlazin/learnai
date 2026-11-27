Implement the write operation for a sliding-window KV cache backed by a fixed-size ring buffer. Once the buffer fills, new entries overwrite the oldest ones in place, except for the first `keep` tokens, which remain permanently. Because wrapping puts slots out of sequence order, your function must also return the cached keys and values in temporal order.

![This problem is the KV cache, here a fixed ring that overwrites its oldest entries.](assets/map.svg)

## Specification

```python
def rotating_write(cache, k_new, v_new, max_size, keep): ...  # returns (cache, keys, values)
```

Let `B` be the batch size, `H` be the number of attention heads, `D` be the head dimension, and `T` be the number of new tokens in the current write.

A **cache** is a tuple `(k, v, offset)`. The buffers `k` and `v` have shape `(B, H, max_size, D)`, and `offset` is a Python `int` tracking the total number of tokens written over the lifetime of the cache. `offset` keeps incrementing after the buffer is full. Number tokens $0, 1, 2, \dots$ in the order they are written.

**What the cache holds.** After $n$ tokens have been written, the cached tokens are:

- the **sinks**, consisting of tokens $0, \dots, \text{keep} - 1$ (whichever have been written so far), and
- the **window**, consisting of the most recent $W = \text{max\_size} - \text{keep}$ non-sink tokens, meaning every token $t$ with $t \ge \text{keep}$ and $n - W \le t < n$.

Until $n$ exceeds `max_size`, this is simply every token written so far.

**Where a token lives.** Token $t$ is stored at slot

$$
\operatorname{slot}(t) = \begin{cases}
t & t < \text{keep}, \\
\text{keep} + (t - \text{keep}) \bmod W & t \ge \text{keep}.
\end{cases}
$$

Sinks occupy slots $0$ through $\text{keep} - 1$. The remaining $W$ slots form a ring. Once the buffer is full, the next token wraps to slot `keep` and overwrites the oldest window token, continuing around the ring. A token never moves after it is written, and each write modifies only the slots of the tokens it stores.

![The buffer for `max_size = 8` and `keep = 2` after 11 tokens. The returned keys are the sinks, then the ring read from the outlined slot.](assets/ring-layout.svg)

**Behavior of `rotating_write(cache, k_new, v_new, max_size, keep)`.** Write the `T` new tokens (numbered `offset` through `offset + T - 1`) into their slots, and return `(cache, keys, values)`:

- `cache` is `(k, v, offset + T)`. Both buffers keep shape `(B, H, max_size, D)`, and for every cached token $t$, slot $\operatorname{slot}(t)$ holds token $t$.
- `keys` and `values` contain the cached entries in **temporal order**, from lowest token number to highest (sinks first, then the window from oldest to newest). Their shape is `(B, H, min(offset + T, max_size), D)`, which is the sequence that attention for the newest token runs over.

Edge cases that follow from these rules:

- `T` may be any integer $\ge 1$. If `T` exceeds the window size $W$, earlier tokens in the chunk that are already outside the window when the write finishes are neither cached nor returned (unless they are sinks).
- A chunk may cross the end of the buffer. Its trailing tokens wrap around and continue at slot `keep`.
- Setting `keep = 0` produces a plain ring over all `max_size` slots.
- Unwritten slots contain arbitrary data and are never returned.
- `max_size` always equals axis 2 of `k` and `v`, and `max_size` and `keep` remain constant across calls on the same cache.

**State in, state out.** The caller always continues with the returned `cache` and never reuses the input `cache`. You may therefore update the buffers in place (as in PyTorch or MLX) or return new buffers of the same shape (as in JAX, where arrays are immutable). No state may live outside the returned tuple.

## Shapes

| name | shape | dtype |
|---|---|---|
| `k`, `v` in `cache` | `(B, H, max_size, D)` | float32 |
| `offset` in `cache` | Python `int` | |
| `k_new`, `v_new` | `(B, H, T, D)` | float32 |
| `keys`, `values` | `(B, H, min(offset + T, max_size), D)` | float32 |

## Example

Let `max_size = 5` and `keep = 2`, so the ring covers slots 2, 3, and 4 ($W = 3$). Use `B = 1`, `H = 1`, and `D = 1`, with token $t$ having key $t$ and value $100 + t$. The buffer starts zero-initialized with `offset = 0`.

| write | tokens | key buffer afterwards | `offset` | `keys` returned |
|---|---|---|---|---|
| 1 | 0, 1, 2, 3 | `[0, 1, 2, 3, 0]` | 4 | `[0, 1, 2, 3]` |
| 2 | 4, 5 | `[0, 1, 5, 3, 4]` | 6 | `[0, 1, 3, 4, 5]` |
| 3 | 6, 7 | `[0, 1, 5, 6, 7]` | 8 | `[0, 1, 5, 6, 7]` |
| 4 | 8 | `[0, 1, 8, 6, 7]` | 9 | `[0, 1, 6, 7, 8]` |

- **Write 1** does not fill the buffer. Slot 4 keeps its initial zero, and only 4 entries are returned.
- **Write 2** wraps around the end. Token 4 goes to slot $2 + (4 - 2) \bmod 3 = 4$. Token 5 goes to slot $2 + (5 - 2) \bmod 3 = 2$, overwriting token 2 (the oldest window token). The buffer holds `[0, 1, 5, 3, 4]`, while temporal order is `[0, 1, 3, 4, 5]`.
- **Write 3** writes tokens 6 and 7 to slots 3 and 4. The ring completes a full turn, so slot order and temporal order align again.
- **Write 4** wraps again into slot 2. Sinks 0 and 1 never move.

## Constraints

- $1 \le B, H \le 4$, $1 \le D \le 16$, $1 \le \text{max\_size} \le 8$, $0 \le \text{keep} < \text{max\_size}$, and $1 \le T \le 12$.
- At most 64 tokens are written to a single cache.
- Because entries are copied directly from the inputs, all checks are exact. After each write, tests verify `keys`, `values`, the slot of every cached token in `k` and `v`, buffer shapes, and that `offset` is a Python `int` with the expected value.
- Each write must modify only the slots where it stores tokens. Rebuilding the buffer in temporal order on every write is not a ring and fails the slot check.
- A Python loop over the `T` new tokens is allowed. No loops over batch rows, heads, or buffer slots are needed.

## Why this matters

Long-running streams such as voice assistants or multi-hour agent sessions eventually exceed a model's context budget. A sliding window keeps only the most recent tokens, and backing it with a ring buffer makes eviction $O(1)$ per token because each new entry overwrites the oldest one in place instead of shifting the entire window.

StreamingLLM (Xiao et al., 2023) showed that plain sliding windows collapse as soon as the initial tokens are evicted, because models place heavy attention weight on the first few tokens regardless of their content. Keeping those initial tokens as "attention sinks" preserves generation quality, and `mlx-lm`'s `RotatingKVCache(max_size, keep)` uses this exact buffer layout. Bugs here stay hidden until the buffer wraps. Evicting the sinks degrades output thousands of tokens into a session, and returning the buffer in slot order misaligns causal masks on every multi-token step after a wrap.
