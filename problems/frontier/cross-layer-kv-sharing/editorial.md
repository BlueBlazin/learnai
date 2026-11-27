## The idea

In a standard decoder, every layer stores its own keys and values. As a result, KV memory scales with the total layer count $L$, and every layer must process every prompt token so no later token attends into a missing cache entry.

When layers $s, \dots, L - 1$ (with $s =$ `share_from`) instead attend over the keys and values cached by layer $s - 1$, each upper layer still computes its own queries, output projection, and MLP, and still passes hidden states upward. It simply reads from another layer's cache instead of maintaining its own. Two properties follow directly.

**Memory.** The upper $L - s$ layers store nothing, so the KV cache holds $s$ layers instead of $L$.

**Prefill.** In a forward pass, a layer's hidden state at position $i$ can be used in only two ways. It feeds the next layer at position $i$, and in an ordinary layer it is also projected into a key and value for future positions to attend to. For a reader layer, the second path does not exist, so its output at position $i$ can only affect the logits at position $i$. Because prefill on a $P$-token prompt only needs the logits at the final position $P - 1$, running the upper layers on `h[-1:]` alone is exact, not an approximation. It produces identical logits while cutting prefill work from $L \cdot P$ layer-tokens to $s \cdot P + (L - s)$.

In `decode_step`, we need the logits for the single new position, so all $L$ layers run on that one token. Decoding uses less KV memory rather than fewer layer-tokens.

Cross-layer KV sharing is not a faster way to evaluate a standard model. It defines a distinct function (producing a different top token in the worked example), so the model must be trained with KV sharing enabled.

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

The exact same code works in JAX and MLX because all tensor operations live inside `LayeredLM`. Slicing `h[-1:]` preserves the sequence axis of length 1. Since `lm.layers(..., kv_from=j)` aligns the $T$ input rows with the last $T$ cached positions in layer `j`, passing `h[-1:]` runs the upper layers at position $P - 1$ while attending over all $P$ prompt positions.

## Common mistakes

- **Running the full stack on every prompt token.** Passing all of `h` into the upper layers produces the right logits and cache, but spends $L \cdot P$ layer-tokens. This bug passes every numeric correctness check and only shows up as missing prefill speedup.
- **Giving upper layers their own caches.** Calling `lm.layers(h, s, L, cache)` without `kv_from` evaluates the ordinary unshared model, producing wrong logits and allocating KV caches for all $L$ layers.
- **Skipping the upper layers during prefill.** The upper layers can be skipped on the first $P - 1$ prompt positions, but not on the last one. The first generated token's logits must come from layer $L - 1$, not layer $s - 1$.
- **Reading from the wrong source layer.** Passing `kv_from=0` only works when $s = 1$. The source must be the last ordinary layer, `share_from - 1`.
- **Passing a stale cache in `decode_step`.** Because caches are immutable, the lower layers return a new cache containing the new token. Passing the old `cache` to the upper layers prevents the new token from attending to itself, causing subtle numeric errors.
- **Forgetting the position offset.** Calling `lm.embed([token])` without `offset=cache.length` embeds every decoded token at position `0`.
- **Prefilling one token at a time.** Looping token by token uses the same `layer_tokens` for the lower layers, but takes $s \cdot P$ layer calls instead of $s$. On a GPU, that replaces one large matrix multiplication with $P$ small ones.

## Complexity

Prefill takes $s \cdot P + (L - s)$ layer-tokens in $L$ layer calls. Decoding takes $L$ layer-tokens in $L$ layer calls per token. After $n$ tokens, stored KV size is $2\,s\,n\,d$ values, a factor of $s / L$ relative to an ordinary decoder.

## In the wild

- **YOCO** (*You Only Cache Once: Decoder-Decoder Architectures for Language Models*, arXiv 2405.05254, NeurIPS 2024) splits a decoder into a self-decoder and a stacked cross-decoder. The self-decoder's output is projected into one shared set of keys and values that every cross-decoder layer attends to via cross-attention. This caches keys and values once and allows prefill to exit early on non-final prompt tokens without changing the output.
- **Gemma 3n** ships this mechanism as KV cache sharing. As described in Google's developer guide (*Introducing Gemma 3n: The developer guide*), a middle layer's keys and values are shared directly with all top layers, yielding roughly $2\times$ the prefill performance of Gemma 3 4B.
- **vLLM** supports shared layers in its hybrid KV cache manager, which skips cache allocation for KV-sharing layers so only layers that write keys and values receive cache blocks.

This problem simplifies a few details of deployed architectures. Here, upper layers directly read the self-attention keys and values of layer $s - 1$, matching Gemma 3n. YOCO instead projects shared keys and values from the self-decoder's output and uses efficient attention (windowed or gated retention) in the lower stack. Models with mixed attention types also share per attention type. For example, in Gemma 4 (`Gemma4TextConfig.kv_sharing_modes` in `transformers` 5.23), the final `num_kv_shared_layers` layers each read from the most recent earlier layer of the same attention type (sliding-window or full), which is not always the immediately preceding layer. The prefill cost argument remains the same. Separate work that keeps a standard architecture and drops caches on selected layers, such as *Layer-Condensed KV Cache for Efficient Inference of Large Language Models* (arXiv 2405.10637), is a different design not modeled here.
