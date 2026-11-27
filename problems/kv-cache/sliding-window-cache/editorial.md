## The idea

A ring buffer is simpler to implement when you avoid tracking a mutable write pointer and instead answer two stateless questions from the total token count $n$:

1. **Which tokens are cached after $n$ tokens?** The sinks $0 \dots \min(\text{keep}, n) - 1$, followed by the last $W = \text{max\_size} - \text{keep}$ non-sink tokens, meaning $\max(\text{keep}, n - W) \le t < n$.
2. **Where does token $t$ live?** At slot $t$ if $t < \text{keep}$, and otherwise at slot $\text{keep} + (t - \text{keep}) \bmod W$.

Because both answers depend only on token indices and `offset`, there is no separate pointer or "has wrapped" flag to keep in sync. `offset` is the entire metadata state.

This layout stays consistent because two window tokens $t$ and $t + W$ map to the same slot, and they can never both be live in a window of size $W$. Writing token $t + W$ automatically overwrites the exact token that just left the window.

With this view, writing and reading are a single scatter/gather pair. A write scatters the newly arrived tokens that survive in the cache into their target slots. A read gathers the slots of all live tokens in increasing token-number order, which gives temporal order directly. The same three lines handle partial buffers, exact fills, wraps, chunks crossing the end of the buffer, and chunks longer than $W$.

## Reference solution (PyTorch)

```python
def rotating_write(cache, k_new, v_new, max_size, keep):
    k, v, offset = cache
    window = max_size - keep
    end = offset + k_new.shape[2]
    sinks = torch.arange(min(keep, end))
    recent = torch.arange(max(keep, end - window), max(keep, end))
    live = torch.cat((sinks, recent))  # cached token numbers, oldest first
    slots = torch.where(live < keep, live, keep + (live - keep) % window)
    fresh = int((live >= offset).sum())  # the new ones are the tail of live
    k[:, :, slots[-fresh:]] = k_new[:, :, live[-fresh:] - offset]
    v[:, :, slots[-fresh:]] = v_new[:, :, live[-fresh:] - offset]
    return (k, v, end), k[:, :, slots], v[:, :, slots]
```

Using `max(keep, end)` as the upper bound of `recent` keeps that range empty while only sink tokens exist. `fresh` is always at least 1 because the newest token is always cached. Filtering down to `live` before scattering also guarantees every index in `slots[-fresh:]` is unique, avoiding undefined overwrite order when a chunk is longer than $W$.

The MLX implementation is identical using `mx` operations. The JAX implementation replaces the two slice assignments with `k = k.at[:, :, slots[-fresh:]].set(...)` and returns the updated arrays. Because callers never reuse the input cache, in-place and functional updates behave identically from the outside.

## Common mistakes

- **Evicting the sinks.** Using `t % max_size` over the whole buffer ignores `keep`. This passes when `keep = 0` and for any `keep` until the buffer first fills.
- **Returning slot order.** After a wrap, the buffer stores sinks followed by newest tokens and then oldest tokens. Returning it directly gives the right set of keys in the wrong order, which breaks causal masking on multi-token steps and any code that assumes the newest token is last.
- **Rolling the entire buffer.** Calling `roll(k, -oldest)` fixes a plain ring, but with `keep > 0` it shifts the sinks into the middle of the sequence. Keep the sinks fixed and roll only the ring slice.
- **Not splitting writes at the wrap boundary.** Checking whether the pointer hit `max_size` only between calls works for single-token writes, but fails when a multi-token chunk crosses the end of the buffer and needs to continue at slot `keep`.
- **Clamping `offset` to `max_size`.** Treating `offset` as the current cache length loses the ring position once the buffer is full, causing subsequent writes to reuse the same slots.
- **Shifting instead of rotating.** Concatenating new entries and slicing the tail keeps the buffer in temporal order and returns the right keys, but copies the entire window on every write and fails the ring slot check.

## Complexity

Writing costs $O(B H \min(T, \text{max\_size}) D)$ for the scatter, independent of total sequence length. Gathering the ordered output costs $O(B H \text{max\_size} D)$ per call. Production engines skip this gather when order does not matter. Memory is fixed at $2 B H \text{max\_size} D$ floats.

## In the wild

`mlx-lm`'s `RotatingKVCache(max_size, keep)` uses this buffer layout with an explicit write index `_idx`. On a single-token decode step, it resets `_idx` to `keep` when `_idx == max_size` and writes via slice assignment. Two details differ from this problem. First, on the single-token path it returns the buffer in slot order without reordering, because a single query attending to all cached keys does not depend on key order. Second, on a multi-token chunk it first reorders the buffer into temporal order, trims, and appends, allowing the buffer to temporarily grow to `max_size + T - 1` so earlier tokens in the chunk can still attend to entries that later tokens in the chunk will evict.

Attention sinks were introduced in StreamingLLM (Xiao et al., 2023, *Efficient Streaming Language Models with Attention Sinks*), which retains 4 initial tokens alongside a sliding window of recent tokens. StreamingLLM also assigns positional embeddings based on a token's position inside the cache rather than its absolute sequence index, keeping positions within the model's training range. A storage layer alone does not do that. If keys are rotated using their original positions (as in `cached-rope-offsets`), this cache bounds memory while positions still grow without limit.
