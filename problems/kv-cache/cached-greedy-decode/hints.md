Consider what a single call to `lm` returns. `logits[-1]` scores the token that comes *after* the last token you passed in. Once the prompt passes through the model, you already have the first generated token without making another call. At that point, which tokens has the model not seen yet, and which one does it need before you can pick the second token?

---

Split the work into prefill and decode. Call `lm(prompt)` once and save the returned `cache`. In each decode step, take the argmax of `logits[-1]`, append it to your result, and pass only that single token with the cache by calling `lm([token], cache)`. Always replace your `cache` variable with the new cache returned by the call, because the previous cache is unchanged and does not include the newest token.

---

Place the stop check between choosing a token and feeding it to the model:

```
logits, cache = lm(prompt)
loop:
    token = argmax(logits[-1])          # as a Python int
    append token to the result
    if token == eos_id or the result has max_new_tokens tokens: return the result
    logits, cache = lm([token], cache)
```

This guarantees that the final token is returned without being fed back into `lm`, matching the exact limits of $g$ calls and $P + g - 1$ tokens processed for a prompt of length $P$ and $g$ generated tokens.
