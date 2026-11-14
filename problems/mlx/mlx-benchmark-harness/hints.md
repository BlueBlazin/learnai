When `model(tokens)` returns, what has been computed? If the clock is read right
after, what did the interval contain? And when the result is finally evaluated, in
which interval does that land? Answer the same three questions for a warm-up run.

---

Write one function that performs a complete run and returns its two times and its
tokens, and use it for warm-up and measured runs alike. Inside it, each phase ends
with an evaluation *before* the clock is read: `mx.eval(token)` blocks until the token
is computed, and the token depends on the logits and the cache, so nothing of the
phase is left pending. In the generation phase evaluate once per step, feeding
`token.reshape(1)` back with the cache. Keep the tokens as arrays while timing and
convert them to ints after the last clock reading.

---

```python
def run():
    start = clock()
    logits, cache = model(prompt_tokens)
    token = mx.argmax(logits[-1])
    mx.eval(token)
    first_token = clock()
    tokens = [token]
    for _ in range(n_tokens):
        logits, cache = model(token.reshape(1), cache)
        token = mx.argmax(logits[-1])
        mx.eval(token)
        tokens.append(token)
    end = clock()
    return first_token - start, end - first_token, tokens
```

Call it `n_warmup` times and drop the results, then `n_runs` times. Use
`statistics.median` on the prompt times and on the generation times separately (it
averages the two middle values of an even count), and divide `len(prompt)` and
`n_tokens` by them. The generation rate counts `n_tokens` steps, not the
`n_tokens + 1` tokens of the run: the first token belongs to the prompt time.
