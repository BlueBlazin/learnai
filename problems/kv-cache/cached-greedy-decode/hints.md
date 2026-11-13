Look at what one call gives you: `logits[-1]` scores the token that comes *after* the
last token you passed in. After the prompt has gone through the model, you already
know the first generated token without any further call. Which tokens has the model
not yet seen at that point, and which of them does it need to see before you can pick
the second token?

---

Prefill, then decode. Call `lm(prompt)` once and keep the cache. Each step then takes
the argmax of the last row of logits, appends it to the result, and passes only that
one token together with the cache: `lm([token], cache)`. Always keep the cache that
the call returns; the one you passed in is unchanged and no longer covers the newest
token.

---

Order the loop body so that the stop check comes between choosing a token and feeding
it:

```
logits, cache = lm(prompt)
loop:
    token = argmax(logits[-1])          # as a Python int
    append token to the result
    if token == eos_id or the result has max_new_tokens tokens: return the result
    logits, cache = lm([token], cache)
```

A token that ends generation is returned but never fed, which is where the limits of
$g$ calls and $P + g - 1$ tokens come from.
