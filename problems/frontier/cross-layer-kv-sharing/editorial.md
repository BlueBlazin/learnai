## The idea

In a standard decoder every layer keeps its own keys and values, so KV memory grows
with the number of layers, and every layer must process every prompt token: a layer
that skipped a position would leave a hole in its cache that later tokens attend
into.

Now let the layers from $s$ up attend over the keys and values of layer $s - 1$
instead of their own. Each still computes its own queries, has its own output
projection and MLP, and passes its state up; it just looks things up in a table
another layer wrote. Two things follow.

**Memory.** The upper layers store nothing. The cache holds $s$ layers instead of
$L$.

**Prefill.** Consider what is ever read from a forward pass. The state of layer
$\ell$ at position $i$ feeds layer $\ell + 1$ at the same position, and, in an
ordinary layer, it is also projected to a key and a value that later positions
attend to. For a reading layer the second use does not exist. Its state at position
$i$ only flows upward at position $i$, to the logits of position $i$. For a prompt
position other than the last, nobody reads those logits. So the upper layers need
to run on exactly one prompt position, the last, and the result is not an
approximation: the logits are the same numbers as if the whole stack had run on
every token. The prefill costs $s \cdot P + (L - s)$ layer-tokens instead of
$L \cdot P$.

A decode step has one new position whose logits are wanted, so it runs all $L$
layers on it: decoding is not faster, only smaller.

The sharing model is not a cheaper way to evaluate the ordinary model. It is a
different function (in the example the two predict different tokens), and a model
has to be trained with the sharing in place.

## Reference solution (PyTorch)

```python
def prefill_shared(lm, prompt, share_from):
    h = lm.embed(prompt)
    h, cache = lm.layers(h, 0, share_from)
    h, cache = lm.layers(h[-1:], share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], cache


def decode_step(lm, token, cache, share_from):
    h = lm.embed([token], offset=cache.length)
    h, cache = lm.layers(h, 0, share_from, cache)
    h, cache = lm.layers(h, share_from, lm.n_layers, cache, kv_from=share_from - 1)
    return lm.head(lm.norm(h))[0], cache
```

The solution is the same text in JAX and MLX: all tensor work is inside the kit, and
the problem is which layers run on which rows. `h[-1:]` keeps the row axis, and the
kit takes the rows handed to a reader to be the last positions the source layer has
cached, so the single row is position $P - 1$ and attends over all $P$ positions.

## Common mistakes

- **The full stack on every prompt token.** Passing all of `h` to the upper layers
  gives the right logits and the right cache, and $L \cdot P$ layer-tokens. This is
  the mistake that survives every correctness test and only shows up as a prefill
  that is no faster than before.
- **Upper layers with caches of their own.** Calling `lm.layers(h, s, L, cache)`
  without `kv_from` runs the ordinary model: different logits, and keys and values
  stored for all $L$ layers. In an engine this is the allocator giving every layer
  a cache because every layer used to need one.
- **Skipping the upper layers in the prefill altogether.** "The prompt only needs
  the lower layers" is true for $P - 1$ positions. The last one produces the first
  generated token, and its logits come from the top of the stack, not from the
  state after layer $s - 1$.
- **Reading the wrong layer.** `kv_from=0` is right only for $s = 1$. The source is
  the last ordinary layer, $s - 1$.
- **A stale cache in the decode step.** Caches are immutable, so the lower layers
  return a new one. Handing the upper layers the old one makes them read layer
  $s - 1$ without the new position: the token cannot attend to itself, and the
  logits are slightly, silently wrong.
- **Forgetting the position offset.** `lm.embed([token])` embeds at position 0.
- **Prefill token by token.** The layer-tokens are the same, but one call per token
  is $s \cdot P$ layer calls where one batched call is $s$; on a GPU that is the
  difference between one matrix product and $P$ small ones.

## Complexity

Prefill: $s \cdot P + (L - s)$ layer-tokens in $L$ layer calls. Decode: $L$
layer-tokens per token. Stored KV: $2\,s\,n\,d$ numbers after $n$ tokens, a factor
$s / L$ of the ordinary model's.

## In the wild

- **YOCO** ("You Only Cache Once: Decoder-Decoder Architectures for Language
  Models", arXiv 2405.05254, NeurIPS 2024) splits a decoder into a self-decoder and
  a cross-decoder stacked on top of it. The self-decoder's output is turned into
  one global set of keys and values that every cross-decoder layer reuses through
  cross-attention, so the pairs are cached once, and the abstract states the
  consequence used here: the prefill can exit early without changing the final
  output.
- **Gemma 3n** ships the idea under the name KV cache sharing. Google's developer
  guide ("Introducing Gemma 3n: The developer guide") describes the keys and values
  of a middle layer being shared directly with all the top layers, and reports
  about twice the prefill performance of Gemma 3 4B.
- **vLLM** accounts for such models in its KV cache manager: according to its
  design document on the hybrid KV cache manager, layers with KV sharing are
  ignored when cache is allocated, and only layers that need a cache get one.
- **What this problem simplifies.** The kit's upper layers read the self-attention
  keys and values of layer $s - 1$ as they are, which is the Gemma 3n arrangement.
  YOCO instead computes the shared keys and values from the self-decoder's *output*
  with projections of their own, and its self-decoder uses efficient (windowed or
  gated-retention) attention. Real models also apply the sharing per attention type.
  In Gemma 4 as `transformers` 5.23 implements it (`Gemma4TextConfig.kv_sharing_modes`,
  read in the code), the last `num_kv_shared_layers` layers reuse, and each reads the
  keys and values stored by the last earlier layer of its own type, sliding-window or
  full, which is not always the layer just below the shared block. The cost
  argument is the same in all of them. Related work that keeps the standard
  architecture and caches only a few layers' pairs, such as "Layer-Condensed KV
  Cache for Efficient Inference of Large Language Models" (arXiv 2405.10637), is a
  different design and is not what is modeled here.
