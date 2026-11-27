## The idea

Split the cache into fixed storage and a single piece of bookkeeping. The storage consists of two buffers allocated once at capacity `C`. The bookkeeping is an integer `length` governed by one invariant:

> Slots `0 .. length - 1` hold the tokens written so far, in order. Nothing may depend on any other slot.

`cache_write` maintains this invariant by writing the next `T` tokens into slots `length .. length + T - 1` and advancing `length` by `T`. `cache_attend` relies on it by reading only slots below `length`. Every common bug in this problem breaks one of those two rules.

Preallocation avoids the quadratic copy cost of concatenation. Growing a cache by concatenating on every decode step allocates a new buffer and copies all $S$ cached tokens each time, moving $O(S)$ bytes per token and $O(n^2)$ bytes across an $n$-token generation, while also changing the tensor shape at every step. Writing into a preallocated buffer moves only $O(T)$ bytes and keeps every buffer shape fixed.

## Reference solution (PyTorch)

```python
def cache_init(capacity, batch, heads, dim):
    shape = (batch, heads, capacity, dim)
    return torch.zeros(shape), torch.zeros(shape), 0


def cache_write(cache, k_new, v_new):
    k, v, length = cache
    end = length + k_new.shape[2]
    if end > k.shape[2]:
        raise ValueError("the cache is full")
    k[:, :, length:end] = k_new
    v[:, :, length:end] = v_new
    return k, v, end


def cache_attend(q, cache):
    k, v, length = cache
    new = q.shape[2]
    k, v = k[:, :, :length], v[:, :, :length]  # views of the valid prefix
    visible = torch.arange(length)[None, :] <= (length - new) + torch.arange(new)[:, None]
    return ops.attention(q, k, v, visible)
```

The causal mask matches `kv-cache-append` with $S = \text{length} - T$ prior tokens, since the `T` queries correspond to the last `T` rows of a full causal attention matrix.

**The three frameworks.** The contract says a cache goes in, a cache comes out, and the caller never reuses the old one. Each framework implements that idiomatically. PyTorch and MLX assign directly to a slice of the existing buffer. JAX arrays are immutable, so the JAX reference writes `k = k.at[:, :, length:end].set(k_new)`, which returns a new array of the same shape. In eager JAX that copies the buffer, but inside `jit`, XLA sees that the old buffer is never used again and performs the update in place. The "never reuse the old cache" rule makes the in-place and functional implementations indistinguishable.

## Common mistakes

- **Attending over the whole buffer.** A zero key has score 0, not $-\infty$, so each unused slot takes a share of the softmax weight and pulls the output toward its zero value. The error shrinks as the buffer fills up, making it easy to miss.
- **Masking only the scores.** Setting unused scores to $-\infty$ gives them weight 0, but the weighted sum still multiplies that 0 by the unused tail of `v`. For finite garbage that happens to work, but for NaN or infinity, $0 \times \text{NaN} = \text{NaN}$ poisons the output. Slicing avoids touching the tail altogether. A masked implementation must sanitize `v` as well.
- **Concatenating in `cache_write`.** Calling `cat(k[:, :, :length], k_new)` produces the right attention outputs but breaks the cache contract. The buffer shrinks to `length` and reallocates on every step.
- **Advancing by `length + 1`.** This works for single-token decode steps, but after a multi-token prefill the next write overwrites live entries.
- **Using `>=` in the capacity check.** Filling the last slot (`length + T == capacity`) is valid. Rejecting it wastes a slot and fails right at the context limit.
- **Truncating instead of raising.** Writing only what fits silently drops tokens while the caller assumes they were cached.
- **Omitting the causal mask among new queries.** A single new token needs no mask, but a multi-token chunk does, just as in `kv-cache-append`.

## Complexity

`cache_write` takes $O(B\,H\,T\,D)$ time and no extra memory when done in place, regardless of how many tokens are already cached. `cache_attend` takes $O(B\,H\,T\,\text{length}\,D)$ time. Memory usage is $2\,B\,H\,C\,D$ floats from the start, whether or not all slots are used, because capacity must be chosen up front.

## In the wild

Hugging Face Transformers' `StaticCache` uses this design. Each layer lazily allocates `torch.zeros((batch, heads, max_cache_len, head_dim))`, and `update` writes new entries with `index_copy_` at positions `arange(T) + cumulative_length`. It returns the full buffer and relies on the attention mask to hide unused slots. Masking only the scores is safe there because the buffer is zero-initialized and never contains NaN. Fixed shapes allow `torch.compile` to reuse the same compiled graph on every decode step.

`mlx-lm`'s `KVCache` grows in blocks of 256 slots, writes with `self.keys[..., prev : self.offset, :] = keys`, and returns the active prefix `self.keys[..., : self.offset, :]`, matching the slicing approach above. `llama.cpp` goes further and allocates the KV buffers for the entire context window when the context is created.

Slicing passes a tensor whose shape changes with `length` into attention. That is fine in eager execution, but triggers a recompilation on every step under `jax.jit` or `torch.compile`. Keeping shapes static all the way through attention is covered in `cuda-static-kv-cache`, and growing the buffer in chunks with MLX's lazy evaluation is covered in `mlx-preallocated-kv-cache`.
