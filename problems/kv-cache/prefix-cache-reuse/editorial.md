## The idea

In a causal transformer, the keys and values at position $i$ depend only on tokens $0$ through $i$, never on later tokens. Therefore, two prompts that agree on their first $n$ tokens produce identical keys and values at positions $0$ through $n - 1$. A KV cache computed for one prompt is valid for the other up to their common prefix, and invalid from the first differing token onward.

This leads to three steps for a prompt of length $P$:

1. **Match.** Compare `prompt` against every key in `store` and find the longest common prefix length $m$.
2. **Truncate.** The matched cache may extend past $m$ tokens if the earlier prompt continued differently. Calling `truncate` trims the cache back to the shared prefix. Without truncation, the leftover positions remain in context and the model attends to the old suffix before the new one.
3. **Prefill the remainder.** Feed the remaining prompt tokens in one call on top of the truncated cache, then use `logits[-1]` to start greedy decoding.

One edge case needs care. If `prompt` equals a stored key or is a prefix of one, then $m = P$ and no suffix remains. Because a KV cache stores keys and values rather than logits, the model must still run on the final prompt token to produce logits for the first generated token. Capping the reused length at $r = \min(m, P - 1)$ handles both this case and single-token prompts ($r = 0$).

Save the cache immediately after the prefill call under `tuple(prompt)` so the stored entry covers only the prompt. Because `TinyLM` returns a fresh cache on every call, subsequent decode steps do not alter the stored cache.

## Reference solution (PyTorch)

```python
def common_prefix_len(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def generate_with_store(lm, store, prompt, max_new_tokens):
    best_key, best = None, 0
    for key in store:
        shared = common_prefix_len(key, prompt)
        if shared > best:
            best_key, best = key, shared
    reused = min(best, len(prompt) - 1)

    if reused > 0:
        cache = store[best_key].truncate(reused)
        logits, cache = lm(prompt[reused:], cache)
    else:
        logits, cache = lm(prompt)
    new_store = {**store, tuple(prompt): cache}

    generated = []
    while True:
        token = int(torch.argmax(logits[-1]))
        generated.append(token)
        if len(generated) == max_new_tokens:
            return generated, new_store
        logits, cache = lm([token], cache)
```

The cost limits are tight because the model must process every prompt token not covered by the store, the store covers at most $m$ tokens, and at least one prompt token must be evaluated to obtain logits. For $g = \text{max\_new\_tokens}$, that requires at least $P - \min(m, P - 1)$ prompt tokens plus the $g - 1$ generated tokens that later steps condition on.

## Common mistakes

- **Reusing only on an exact match (`if tuple(prompt) in store`).** A direct dictionary lookup misses almost all practical hits, including follow-up conversation turns and new questions sharing the same system prompt.
- **Reusing only whole entries.** Requiring a stored key to be a complete prefix of `prompt` handles growing conversations, but misses branching prompts that share a system prompt and then diverge.
- **Forgetting to truncate.** Continuing directly from a stored cache works when the entire stored key is a prefix of `prompt`, so simple tests pass. When the stored key diverges, token counts look right but the model attends to the stale suffix and outputs wrong tokens.
- **Feeding an empty list on a full match.** Setting $r = m = P$ passes `prompt[P:]` (an empty list) to `lm`, which fails. Truncating to $P$ without making a prefill call also leaves no logits for the decode loop.
- **Always holding one token back ($r = m - 1$).** Subtracting 1 unconditionally avoids the empty-call bug, but wastes one extra token on every non-exact match.
- **Taking the first matching key instead of the longest.** This produces the right tokens, but wastes prefill work depending on dictionary iteration order.
- **Storing the wrong cache or dropping old entries.** Saving the post-generation cache under `tuple(prompt)` breaks the invariant that each cache matches its key, causing future requests to condition on the old reply. Returning a dict without the existing `store` entries discards other cached prefixes.

## Complexity

Prefix matching takes $O(\sum_{\text{keys}} \min(|\text{key}|, P))$ token comparisons. Model evaluation processes $P - r + g - 1$ tokens across $g$ forward calls, and the store grows by one cache of length $P$. The linear scan over keys does not scale to large stores, so production servers index stored prefixes instead.

## In the wild

- `llama.cpp` tracks the token sequence of the last prompt in each slot. With `cache_prompt` enabled, it finds the common prefix with the new prompt, keeps those KV entries, discards the rest, and evaluates only the new suffix.
- `mlx-lm` maintains an LRU store of prompt caches and looks up the closest prefix. When a stored cache is longer than the match, it trims a copy and caps the reused prefix at `len(tokens) - 1`, using the same rule as this problem. It also stores the generated reply in the key so the next turn reuses both the prompt and the previous answer, whereas this problem stores only the prompt.
- vLLM's automatic prefix caching avoids scanning prompts. It stores the KV cache in fixed-size blocks, hashes each full block together with all preceding blocks, and reuses every leading block whose hash already exists. `block-hash-prefix-cache` builds that index, and `radix-prefix-cache` builds the radix-tree index used by SGLang.
- Cheap truncation depends on standard causal attention. Models with sliding windows, recurrent states, or evicted entries cannot always roll back to an arbitrary length, so engines fall back to checkpoints or recomputation.
