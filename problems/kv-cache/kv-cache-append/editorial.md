## The idea

In a causal model, the output at position $p$ is a function of the query at $p$ and
of the keys and values at positions $0..p$. Nothing about positions before $p$ depends
on what comes later, so their keys and values can be computed once and kept. A step
with a cache is then the bottom rows of the full attention computation: the rows of
the new tokens, against all columns.

Seen that way, the mask needs no new rule. In the full $(S+T) \times (S+T)$ causal
mask, row $p$ allows columns $j \le p$. The step keeps rows $p = S, \dots, S + T - 1$,
so row $i$ of the step allows $j \le S + i$: a block of $S$ fully visible cached
columns, then a $T \times T$ causal triangle. Two special cases explain most of the
bugs. With $S = 0$ the block is empty and the mask is the familiar triangle. With
$T = 1$ the single row allows everything, so a decode step works without any mask,
and code that was only ever tested by decoding has never run its mask.

Appending along the sequence axis with the old entries first makes "column index"
and "absolute position" the same number, which is what lets one comparison of two
`arange`s express the rule.

## Reference solution (PyTorch)

```python
def attend_step(q_new, k_new, v_new, cache):
    if cache is None:
        k, v = k_new, v_new
    else:
        k = torch.cat((cache[0], k_new), dim=2)
        v = torch.cat((cache[1], v_new), dim=2)
    new, total = q_new.shape[2], k.shape[2]
    past = total - new
    visible = torch.arange(total)[None, :] <= past + torch.arange(new)[:, None]  # (T, S + T)
    return ops.attention(q_new, k, v, visible), (k, v)
```

`torch.cat` allocates new tensors, so the cache that came in is untouched, and the
function is a pure map from (inputs, state) to (output, state). The JAX and MLX
versions are the same lines with `concatenate` and `axis=2`. This is also why the
interface is a tuple and a return value instead of an object that grows: a JAX array
cannot be appended to in place at all, and an MLX array is a node of a lazy graph, so
"the same buffer, now longer" is not something either framework offers.

## Common mistakes

- **No mask in the step.** `softmax(q K^T) V` is exact for one new token and wrong
  for a chunk: each new token also sees the new tokens after it. The outputs then
  depend on the chunking.
- **A mask that forgets the offset.** Reusing the square-mask code as `j <= i` on a
  `(T, S + T)` matrix hides most of the cache: token `i` sees only the first `i + 1`
  entries. Correct with no cache, wrong with one.
- **Attending over the cache only, appending afterwards.** The new token never sees
  itself (or the earlier tokens of its chunk). The order of the two operations is the
  whole point: append, then attend.
- **Appending on the wrong axis.** In the `(B, H, T, D)` layout the sequence is axis
  2. Code ported from a `(B, T, H, D)` layout concatenates on axis 1 and stacks heads;
  it usually crashes on a shape mismatch, and when the cache and the chunk have equal
  lengths it does not.
- **New entries before old ones.** The cache is then not in sequence order, and every
  later mask and position computation that assumes "index = position" is wrong.
- **Returning the cache that came in.** The outputs of this step are right and the
  next step has lost these tokens.

## Complexity

One step costs $O(B\,H\,T\,(S+T)\,D)$ time for the scores and the mix. Feeding $n$
tokens one at a time therefore costs $O(n^2)$ in total, the same order as one full
pass: the cache does not make attention itself cheaper, it removes the recomputation
of every earlier token's keys, values, and MLP in every layer.

The concatenation copies the whole cache, $O(S)$ per step and $O(n^2)$ over a
generation in copied memory. That copy is the motivation for the next problems in
this track, which write into a preallocated buffer.

## In the wild

Hugging Face Transformers' default `DynamicCache` is this function's cache: each
layer's update is `torch.cat([self.keys, key_states], dim=-2)`. Its `StaticCache`
preallocates instead. mlx-lm's `KVCache.update_and_fetch(keys, values)` has the same
shape of interface as `attend_step` (hand in the new entries, get back everything to
attend over), with a buffer that grows in steps of 256 tokens. In all of them the
causal mask for a multi-token step is built by the caller with the cache length as
an offset, which is exactly the `(T, S + T)` mask of this problem.
