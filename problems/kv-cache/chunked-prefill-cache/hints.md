What does the model need to treat the second chunk as a continuation of the first
rather than the start of a new sequence? Among all the logits rows returned across
calls, which single row scores the token *after* the prompt?

---

Step through `prompt` in increments of `chunk_size` and slice each chunk as
`prompt[start : start + chunk_size]`. Python slicing stops at the end of the list, so
the last chunk automatically contains the remaining tokens. Initialize `cache = None`,
pass it into every call, and save the returned cache for the next iteration. Check
that your loop never produces an empty slice when `len(prompt)` is a multiple of
`chunk_size`.

---

```
cache = None
for start in 0, chunk_size, 2 * chunk_size, ... while start < len(prompt):
    logits, cache = lm(prompt[start : start + chunk_size], cache)
return logits[-1], cache
```

`lm(tokens, None)` behaves the same as `lm(tokens)`, so the first chunk needs no
special case. After the loop finishes, `logits` holds the outputs of the final chunk,
and `logits[-1]` is the answer. In Python, `range(0, len(prompt), chunk_size)` yields
exactly $\lceil P / c \rceil$ non-empty start indices.
