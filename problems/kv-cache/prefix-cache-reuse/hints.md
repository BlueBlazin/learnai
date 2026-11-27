In a causal model, the keys and values for the first $n$ tokens depend only on those $n$ tokens and never on what follows. A cached prompt is therefore valid for a new prompt up to the first position where they differ. How can you find that position, and what must you do with any cached tokens beyond it?

---

Write a helper that counts the leading matching tokens between two sequences, and run it against every key in `store` to find the maximum match length $m$ and its key. You can reuse up to $m$ tokens, except when $m$ equals the prompt length $P$, because `lm` requires at least one token and you need logits for the last prompt token. Cap the reused length at $r = \min(m, P - 1)$, then call `store[best_key].truncate(r)` to get a cache covering exactly the first $r$ tokens.

---

```python
r = min(longest_common_prefix, len(prompt) - 1)
if r > 0:
    logits, cache = lm(prompt[r:], store[best_key].truncate(r))
else:
    logits, cache = lm(prompt)
new_store = {**store, tuple(prompt): cache}  # covers exactly the prompt
```

Capture the cache for `new_store` right after the prefill call, before starting the decode loop. The decode loop is the same as in `cached-greedy-decode` without a stop token, and because every `lm` call returns a new cache, decoding will not mutate the stored entry.
