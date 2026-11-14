## The idea

RoPE turns pair $i$ of a vector at position $p$ by the angle $p\,\theta_i$. For a
rotated query at position $m$ and a rotated key at position $n$, the rotations combine
pair by pair, and the score depends on the positions only through $n - m$:

$$
\langle \operatorname{rope}(q, m),\ \operatorname{rope}(k, n) \rangle
= \langle q,\ \operatorname{rope}(k, n - m) \rangle .
$$

Two consequences decide everything in this problem.

- **A key can be rotated once and stored.** The distance between token 5 and token 9
  is 4 now and after a thousand more tokens. So a key rotated for its absolute
  position when it was new is still the right key for every later query, provided
  that every later query is rotated for *its* absolute position too. The cache holds
  rotated keys, and a step rotates only what is new.
- **Positions must be absolute across calls.** Within one call, any common offset
  cancels: positions `[0, 1]` and `[37, 38]` give the same scores between the two new
  tokens. Against the cache it does not cancel: the cached keys were rotated for
  positions `0..36`, and a query rotated for position 0 is at distance $-36$ to $0$
  from them instead of $1$ to $37$.

The position of a new token is the number of tokens before it in its own row, which
is the row's cache length plus its index in the chunk. That is a property of the row,
not of the batch.

## Reference solution (PyTorch)

```python
def rope_positions(cache_lengths, n_new):
    return cache_lengths[:, None] + torch.arange(n_new)  # (B, 1) + (n_new,) -> (B, n_new)


def rotate_new(q, k, positions, theta):
    return ops.rope(q, positions, theta), ops.rope(k, positions, theta)
```

Both are pure functions of their arguments, so the JAX and MLX versions are the same
two lines with `jnp.arange` and `mx.arange`. The state (how long each row's cache is)
lives with the caller, next to the cache itself.

## Common mistakes

- **Positions that restart at 0 in every call.** `arange(T)` is right for a prefill
  from an empty cache and wrong for every step after it. Tests that only run full
  prompts never see it.
- **Off by one.** A row with `n` cached tokens has used positions `0..n-1`; the next
  position is `n`, not `n + 1`. The scores among the new tokens are unaffected, so
  this too is visible only against the cache, as a gap of one position.
- **One offset for the whole batch.** Using `cache_lengths[0]`, the maximum, or a
  scalar "current length" is right when every row has the same length and wrong
  whenever prompts differ in length or a request joins a running batch.
- **Rotating every row with the first row's positions.** The same mistake one level
  down: code written for a `(T,)` position vector, applied to a `(B, T)` table by
  taking `positions[0]`.
- **Rotating the queries only.** "Keys in the cache are already rotated" is true of
  the cached keys. The new keys arrive unrotated and are rotated in this step, before
  they are appended.
- **Rotating the cached keys again.** The opposite error, made one level up in the
  attention layer: concatenating first and then applying RoPE to all the keys. Every
  cached key is turned once more per step. The interface of this problem makes that
  impossible by never showing `rotate_new` the cache; keep it that way in your own
  attention code by rotating before you append.
- **Dropping `theta`.** The base is part of the model, not a constant: 10000 in the
  original RoPE and Llama 2, 500000 in the Llama 3 configuration.

## Complexity

`rope_positions` is $O(B\,n_{\text{new}})$. `rotate_new` is $O(B\,H\,T\,D)$: it is
proportional to the new tokens only and independent of the cache length, which is the
point. Re-rotating the cache each step would cost $O(S)$ per step for nothing.

## In the wild

mlx-lm's attention layers do exactly this, in this order: `queries = self.rope(queries,
offset=cache.offset)`, the same for the keys, and then `cache.update_and_fetch(keys,
values)`, so the cache only ever receives rotated keys. Its offset is a single number
per cache, which is enough when all rows of a batch advance together. Engines that
batch unrelated requests carry positions per token: llama.cpp's `llama_batch` has a
`pos` entry next to every token and sequence id, and Hugging Face models accept
`position_ids` of shape `(B, T)`, which `generate` derives from the attention mask so
that padded rows still count their real tokens.

"Stored rotated" is a design choice with one notable exception. StreamingLLM (Xiao et
al., 2023) evicts tokens from the middle of the cache and then numbers the survivors
by their slot in the cache, not by their position in the text. A key's position
therefore changes as the cache slides, so that system caches keys *before* the rotary
transformation and rotates the cached keys at every step. Stored rotated keys are
correct exactly as long as a token's position never changes.
