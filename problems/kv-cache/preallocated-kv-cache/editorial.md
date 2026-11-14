## The idea

Split the cache into storage and bookkeeping. The storage is two buffers whose shape
is fixed when the cache is created. The bookkeeping is one integer, `length`, and one
invariant:

> slots `0 .. length - 1` hold the tokens written so far, in order; nothing may
> depend on any other slot.

`cache_write` maintains the invariant: it fills the next `T` slots and adds `T` to
the length. `cache_attend` relies on it: it reads slots below the length and nothing
else. Every bug in this problem breaks one half of that sentence.

The reason to bother is cost. Appending by concatenation allocates a new tensor and
copies the old one on every step: $O(S)$ per token, $O(n^2)$ bytes moved over a
generation of $n$ tokens, and a different tensor shape at every step. Writing into a
preallocated buffer moves $O(T)$ bytes and leaves every shape as it was.

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

The mask is the one from `kv-cache-append` with $S = \text{length} - T$: the queries
are the last $T$ rows of the full causal computation.

**The three frameworks.** The interface is "a cache goes in, a cache comes out, and
the old one is not used again", which each framework implements in its own natural
way. PyTorch and MLX assign to a slice of the existing buffer. JAX arrays cannot be
assigned to, so the JAX reference writes `k = k.at[:, :, length:end].set(k_new)`,
which returns an array of the same shape. Run eagerly, that copies the buffer; inside
`jit`, when the caller does not use the old buffer again (which is exactly what the
contract promises), XLA performs the update in place. The "never reuse the old
cache" rule is what makes the in-place and the functional versions indistinguishable.

## Common mistakes

- **Attending over the whole buffer.** A zero key has score 0, not $-\infty$, so each
  unused slot takes a share of the attention and pulls the output toward its (zero)
  value. The error shrinks as the buffer fills up, which makes it easy to miss.
- **Masking only the scores.** Setting the tail's scores to $-\infty$ gives the tail
  weight 0, and then the weighted sum multiplies that 0 by whatever is in the tail of
  `v`. For finite garbage the result is right. For NaN or infinity, $0 \times
  \text{NaN} = \text{NaN}$ and the output is NaN. Slicing avoids the question; a
  masked implementation must sanitize `v` too.
- **Concatenating in `cache_write`.** `cat(k[:, :, :length], k_new)` produces the
  right outputs and the wrong cache: the buffer shrinks to the length and is
  reallocated at every step. It is `kv-cache-append` with extra steps.
- **`length + 1`.** Correct for every decode step and wrong after a prefill of more
  than one token; the next write then lands on top of live entries.
- **`>=` in the capacity check.** `length + T == capacity` fills the last slot and is
  valid. Refusing it wastes a slot and fails exactly at the model's context limit.
- **Truncating instead of raising.** Writing "what fits" drops tokens silently; the
  caller believes they are cached.
- **No causal mask among the new queries.** As in `kv-cache-append`: one new token
  needs no mask, a chunk does.

## Complexity

`cache_write` is $O(B\,H\,T\,D)$ time and no new memory (in place), independent of
how much is cached. `cache_attend` is $O(B\,H\,T\,\text{length}\,D)$. Memory is
$2\,B\,H\,C\,D$ floats from the first call on, whether or not they are used: the
price of preallocation is that the capacity must be chosen in advance.

## In the wild

Hugging Face Transformers' `StaticCache` is this design: each layer lazily allocates
`torch.zeros((batch, heads, max_cache_len, head_dim))`, and `update` writes with
`index_copy_` at positions `arange(T) + cumulative_length`. It returns the whole
buffer and leaves it to the attention mask to hide the unused slots, which works
because the buffer was zero-initialized and is never NaN: the "mask only the scores"
approach is safe exactly when you control what the tail contains. Its purpose is
`torch.compile`: with fixed shapes, the compiled graph is reused at every step.

mlx-lm's `KVCache` grows in blocks of 256 slots, writes with
`self.keys[..., prev : self.offset, :] = keys`, and returns
`self.keys[..., : self.offset, :]`: the slice-the-prefix approach of the reference
solution. llama.cpp goes one step further and allocates the KV buffers for the whole
context window when the context is created.

Slicing gives the attention a tensor whose shape changes with `length`, which is fine
in eager code and a recompilation per step under `jax.jit` or `torch.compile`. Keeping
the shapes static all the way through the attention is the subject of
`cuda-static-kv-cache`; growing the buffer in steps and MLX's lazy evaluation are the
subject of `mlx-preallocated-kv-cache`.
