What does the model need in order to treat the second chunk as a continuation of the
first, and not as the beginning of a new sequence? And of all the logits the calls
return, which single row says anything about the token *after* the prompt?

---

Walk over the prompt in steps of `chunk_size` and slice: the chunk that starts at
`start` is `prompt[start : start + chunk_size]`. A Python slice stops at the end of
the list, so the last chunk is automatically the shorter remainder. Start with
`cache = None`, pass the current cache with every chunk, and keep the cache each call
returns for the next one. Make sure the loop cannot produce an empty slice when the
prompt length is a multiple of `chunk_size`.

---

```
cache = None
for start in 0, chunk_size, 2 * chunk_size, ... while start < len(prompt):
    logits, cache = lm(prompt[start : start + chunk_size], cache)
return logits[-1], cache
```

`lm(tokens, None)` is the same as `lm(tokens)`, so the first chunk needs no special
case. After the loop, `logits` belongs to the final chunk, and its last row is the
answer; `range(0, len(prompt), chunk_size)` yields exactly $\lceil P / c \rceil$
starts.
