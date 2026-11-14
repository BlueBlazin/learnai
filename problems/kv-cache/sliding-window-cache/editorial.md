## The idea

A ring buffer is easiest to get right when the code never asks "where is the write
pointer?" and asks two stateless questions instead:

1. **Which tokens are cached after $n$ tokens?** The sinks $0..\text{keep}-1$ and the
   last $W = \text{max\_size} - \text{keep}$ of the rest: $\max(\text{keep}, n - W)
   \le t < n$.
2. **Where does token $t$ live?** In slot $t$ if it is a sink, else in slot
   $\text{keep} + (t - \text{keep}) \bmod W$.

Both are pure functions of the token number, so there is no pointer to keep in sync
and no "has it wrapped yet" flag. The single integer `offset` is the whole state
besides the buffers.

Why the layout is consistent: two window tokens $t$ and $t + W$ share a slot, and
they are never cached at the same time, since the window is $W$ tokens long. So when
token $t + W$ is written, the entry it replaces is exactly the token that has just
left the window. Eviction is not an operation; it is what overwriting means.

Writing and reading are then the same gather/scatter pattern. A write scatters the
new tokens that are still cached into their slots. A read gathers the slots of the
cached tokens in increasing token number, which *is* temporal order. Not-yet-full,
exactly full, wrapped, a chunk that straddles the end of the buffer, and a chunk
longer than the window all go through the same three lines.

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

`max(keep, end)` keeps the second range empty while only sinks have been written.
`fresh` is at least 1, because the newest token is always cached. No slot appears
twice in `slots`, so the scatter is well defined; that is the reason for dropping the
doomed tokens of an overlong chunk instead of writing them and letting later ones
overwrite them, which an indexed assignment with repeated indices does not promise.

The MLX version is identical with `mx` functions. The JAX version replaces the two
assignments with `k = k.at[:, :, slots[-fresh:]].set(...)`: the same scatter,
returned as a new array. Because the caller never reuses the old cache, the in-place
and the functional versions cannot be told apart from outside.

## Common mistakes

- **Evicting the sinks.** A ring over the whole buffer (`t % max_size`) ignores
  `keep`. It is correct for `keep = 0` and, for any `keep`, correct until the buffer
  first fills up.
- **Returning slot order.** After a wrap the buffer reads `sinks, newest ..., oldest
  ...`. Returning it as it is gives the right *set* of keys in the wrong order. For a
  single new query that sees everything, the order is irrelevant; it matters as soon
  as anything positional is applied to the result: a causal mask for a multi-token
  step, or a slice that assumes the newest token is last.
- **Rolling the whole buffer.** `roll(k, -oldest)` restores the order of a plain
  ring. With sinks it rotates them into the middle of the sequence; the sinks must be
  taken out first and only the ring rolled.
- **Not splitting a write at the wrap point.** A chunk that reaches past the last
  slot continues at slot `keep`. Code that handles the wrap only between writes
  ("if the pointer is at the end, move it to `keep`") is right for single tokens and
  wrong for the first chunk that straddles the end.
- **Treating `offset` as a length.** Clamping it to `max_size` loses the ring
  position: once full, every write computes the same slots.
- **Shifting instead of rotating.** Concatenating and trimming keeps the buffer in
  temporal order and returns the right keys, at the cost of copying the whole window
  on every token. That is the sliding window without the ring.

## Complexity

A write costs $O(B\,H\,\min(T, \text{max\_size})\,D)$ for the scatter, independent of
how many tokens were ever written. The reference also gathers the full cache to
return it in temporal order, $O(B\,H\,\text{max\_size}\,D)$ per call; that copy is
the price of handing back an ordered tensor, and real engines avoid it where the
order does not matter (see below). Memory is fixed at $2\,B\,H\,\text{max\_size}\,D$
floats.

## In the wild

mlx-lm's `RotatingKVCache(max_size, keep)` uses this layout with a write index
`_idx`: on a single-token step, `if self._idx == self.max_size: self._idx = self.keep`
and then a slice assignment at `_idx`. Two differences from this problem are worth
knowing. On that single-token path it returns the buffer in slot order, without
reordering, because one query that may see every entry does not care about the order
of the keys. And for a multi-token chunk it first puts the buffer in temporal order
(`_temporal_order`, the three-piece concatenation from the last hint), trims, and
appends, letting the buffer temporarily exceed its size; its comment gives the
reason: "The largest size is self.max_size + S - 1 to ensure every token gets at
least self.max_size context". In a chunk of $T$ tokens only the last one attends
over exactly the set this problem returns; the earlier ones would want tokens that
the later ones have already evicted.

The attention-sink idea is from StreamingLLM (Xiao et al., 2023, "Efficient
Streaming Language Models with Attention Sinks"), which keeps four initial tokens and
a rolling window of recent ones. It also changes how positions are assigned: tokens
are numbered by their place in the cache, not in the original text, so that the model
never sees a position beyond its training length. That is a second half of the
technique that a storage layer alone does not provide. A cache like this one, which
stores keys already rotated for their original positions (as in
`cached-rope-offsets`), slides the memory but lets positions grow without bound.
