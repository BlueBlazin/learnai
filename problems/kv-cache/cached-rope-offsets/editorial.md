## The idea

For pair $i$ with inverse frequency $\theta_i = \text{theta}^{-2i/D}$, RoPE rotates
a vector at position $p$ by the angle $p\,\theta_i$. When a rotated query at position
$m$ meets a rotated key at position $n$, the rotations combine pair by pair so the dot
product depends only on the relative offset $n - m$:

$$
\langle \operatorname{rope}(q, m),\ \operatorname{rope}(k, n) \rangle
= \langle q,\ \operatorname{rope}(k, n - m) \rangle.
$$

Two rules follow directly from this identity.

- **Rotate each key once and cache it.** The distance between token 5 and token 9
  remains 4 even after a thousand more tokens arrive. A key rotated for its absolute
  position when it enters the model stays valid for every future query, as long as
  each future query is rotated for *its* absolute position too. The cache stores
  rotated keys, and each step rotates only the new tokens.
- **Keep positions absolute across calls.** Within a single call, any shared shift
  cancels out, so positions `[0, 1]` and `[37, 38]` give the same scores between the
  two new tokens. Against the cache, the shift does not cancel. Cached keys were
  rotated for positions `0..36`, so a query rotated for position 0 sits at distances
  $-36$ to $0$ from them instead of $1$ to $37$.

A new token's position is the number of tokens ahead of it in its own row, which is
that row's cache length plus its index in the new chunk. Each row therefore carries
its own offset independently of the rest of the batch.

## Reference solution (PyTorch)

```python
def rope_positions(cache_lengths, n_new):
    return cache_lengths[:, None] + torch.arange(n_new)  # (B, 1) + (n_new,) -> (B, n_new)


def rotate_new(q, k, positions, theta):
    return ops.rope(q, positions, theta), ops.rope(k, positions, theta)
```

Both functions are pure, so the JAX and MLX implementations are identical except for
using `jnp.arange` or `mx.arange`. Each row's cache length lives with the caller next
to the cache itself.

## Common mistakes

- **Restarting positions at 0 on every call.** Using `arange(T)` works for prefill
  from an empty cache and fails on every subsequent decode step. Tests that only run
  full prompts never catch it.
- **Being off by one.** A row with `n` cached tokens already occupies positions
  `0..n-1`, so the next token starts at `n`, not `n + 1`. Scores among the new tokens
  still match, so the bug only appears as a one-position gap against the cache.
- **Using one offset for the whole batch.** Taking `cache_lengths[0]`, the maximum, or
  a single scalar length works only when all rows have the same length. It breaks as
  soon as prompts differ in length or a request joins an active batch.
- **Rotating every row with the first row's positions.** Passing `positions[0]` to a
  rotation helper written for a 1D `(T,)` vector applies row 0's offsets to every row
  in the batch.
- **Rotating only the queries.** Cached keys are already rotated, but the new keys
  arrive unrotated and must be rotated in this step before being appended to the
  cache.
- **Rotating cached keys again.** Concatenating new keys onto the cache before calling
  RoPE turns every cached key an extra time on every step. This problem's signature
  prevents that mistake by hiding the cache from `rotate_new`. Keep the same order in
  full attention layers by rotating before appending.
- **Ignoring `theta`.** The RoPE base is a model parameter, not a fixed constant. The
  original RoPE paper and Llama 2 use `10000`, whereas Llama 3 uses `500000`.

## Complexity

`rope_positions` runs in $O(B\,n_{\text{new}})$ time. `rotate_new` runs in
$O(B\,H\,T\,D)$ time, scaling only with the new tokens and remaining independent of
the cached sequence length $S$. Re-rotating the cache on every step would waste $O(S)$
work per step.

## In the wild

`mlx-lm` attention layers follow this exact sequence: rotate queries with
`self.rope(queries, offset=cache.offset)`, do the same for keys, and then call
`cache.update_and_fetch(keys, values)` so the cache only stores rotated keys. A single
offset per cache suffices when every row in a batch advances together. Engines that
batch independent requests track positions per token. `llama.cpp`'s `llama_batch`
stores a `pos` entry alongside every token and sequence ID, and Hugging Face models
take `position_ids` of shape `(B, T)`, which `generate` computes from the attention
mask so padded rows still count only their real tokens.

Storing rotated keys has one well-known exception. StreamingLLM (Xiao et al., 2023)
evicts tokens from the middle of the cache and re-indexes the surviving tokens by
their cache slot rather than their original text position. Because a key's position
changes as the window slides, StreamingLLM caches unrotated keys and rotates them at
every step. Caching rotated keys is valid whenever a token's position stays fixed.
