With a KV cache, "the longest common prefix" is the amount you can reuse, because the
cache can be cut to any length. A recurrent state has read exactly the tokens of its
key, all of them. For which stored keys is that set of tokens something the new
prompt also starts with? And at which positions of the new sequence will you hold a
state that you could save for the next request?

---

Three separate steps. Pick: the longest key with `len(key) < len(prompt)` and
`tuple(prompt[:len(key)]) == key`. Prefill: a state exists only at the end of a call,
so cut the remaining prompt at the multiples of `interval` and at its end, one call
per piece, saving after each. Generate: every decode step is its own call, so after
each one check whether the sequence length is a multiple of `interval`. Keep the
store as a dict in recency order: to make a key the most recent, remove it and
insert it again.

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

`(done // interval + 1) * interval` is the next multiple strictly above `done`,
whether or not `done` itself is one; a resumed state need not sit on a multiple.
