When `model(tokens)` returns, what has actually been computed? If you read `clock()` right after, what work did that interval measure, and which interval will pay for the computation once it finally runs? Ask the same questions for a warm-up run.

---

Write a single helper function that executes one complete run and returns its prompt time, generation time, and tokens, then use it for both warm-up and measured runs. Inside that helper, end each phase by calling `mx.eval(token)` *before* reading the clock. Because `token` depends on the logits and the cache, evaluating it blocks until all work for that phase finishes. In the generation loop, evaluate once per step and feed `token.reshape(1)` back with `cache`. Keep tokens as MLX arrays while timing and convert them to Python ints after the final clock reading.

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

Call `run()` `n_warmup` times and discard the results, then call it `n_runs` times and keep the timings. Use `statistics.median` separately on the prompt times and generation times (it averages the two middle values when `n_runs` is even). Divide `len(prompt)` by the median prompt time and `n_tokens` by the median generation time, since the first of the `n_tokens + 1` tokens belongs to the prompt phase.
