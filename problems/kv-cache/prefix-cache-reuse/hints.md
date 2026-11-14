Two prompts that start with the same $n$ tokens produce the same keys and values for
those $n$ positions, whatever comes after: a causal model never looks ahead. So a
cache computed for one prompt is valid for another *up to the point where they first
differ*. How do you find that point, and what must happen to the part of the stored
cache that lies beyond it?

---

Write a helper that counts the leading tokens two sequences share. Run it against
every key of the store and remember the key with the largest count $m$. Then decide
how much to reuse: all $m$ tokens, unless that is the whole prompt. A model call
needs at least one token and you need the logits of the last prompt token, so reuse
at most $P - 1$. `store[key].truncate(r)` gives a cache for exactly the first `r`
tokens, whether the entry was longer, diverged, or matched completely.

---

```
r = min(longest common prefix over the keys, len(prompt) - 1)
if r > 0:  logits, cache = lm(prompt[r:], store[best_key].truncate(r))
else:      logits, cache = lm(prompt)
new_store = {**store, tuple(prompt): cache}     # this cache covers exactly the prompt
```

Take the entry for the new store right here, before the decode loop: at this moment
the cache covers the prompt and nothing else. The decode loop is the one from
`cached-greedy-decode` without the stop token; because every call returns a new
cache, continuing to decode does not change the cache you stored.
