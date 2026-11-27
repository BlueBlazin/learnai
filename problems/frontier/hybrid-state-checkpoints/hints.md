With a KV cache, the longest common prefix determines how much you can reuse because you can truncate the cache to any length. A recurrent state, however, has folded in every token of its key. Ask yourself which stored keys consist exclusively of tokens that the new prompt starts with, and at which positions along the new sequence you will actually have a state in hand to save.

---

Break the function into three steps.

1. **Pick a checkpoint.** Find the longest key in `store` with `len(key) < len(prompt)` and `tuple(prompt[:len(key)]) == key`.
2. **Prefill in segments.** Because a state is returned only at the end of a call, split the remaining prompt at multiples of `interval` and at `len(prompt)`, making one call per segment and saving after each.
3. **Decode and checkpoint.** Every decode step is already a single-token call, so check whether `len(sequence)` is a multiple of `interval` after each fed token.

Keep the store as a Python `dict` in recency order. To refresh a key as most recently used, pop it and re-insert it.

---

```
new_store = dict(store)
save(seq, state):  key = tuple(seq); new_store.pop(key, None); new_store[key] = state

best = longest usable key, or None;  state, done = (store[best], len(best)) or (None, 0)
if best: save(best, state)                              # used: now the most recent

while done < P:
    stop = min(P, (done // interval + 1) * interval)    # absolute multiples, then P
    logits, state = lm(prompt[done:stop], state);  done = stop
    save(prompt[:done], state)

decode loop: after feeding a token, if len(sequence) % interval == 0: save(sequence, state)

while len(new_store) > max_checkpoints:
    del new_store[next(iter(new_store))]                # least recently used
```

The expression `(done // interval + 1) * interval` gives the next multiple of `interval` strictly greater than `done`, even when a resumed checkpoint does not sit on a multiple of `interval`.
