`mx.async_eval(x)` returns immediately while the GPU starts computing `x` in the background, and a later `x.item()` waits only for whatever work is not finished yet. What needs to already be in the GPU queue at the moment you block on token $t$? And to build the graph for step $t + 1$, do you need token $t$'s Python integer value or just its MLX array?

---

Keep two steps alive at once: the current step (already scheduled and ready to be read) and the next step (built from the current token **array** via `token.reshape(1)` and then scheduled). Each step calls `logits, cache = model(...)`, computes the log-softmax of `logits[-1]`, takes the argmax token, and indexes its log-probability. Pass both `token` and `logprob` to `mx.async_eval` because `logprob` is computed from `token`, so scheduling `token` alone leaves `logprob` unevaluated. You do not need to pass `cache` because `token` already depends on every array in it.

---

Structure the generator loop so step `index + 1` is scheduled before reading `token`:

```python
token, logprob, cache = step(mx.array(prompt), None)
mx.async_eval(token, logprob)
for index in range(n):
    if index + 1 < n:  # nothing follows the last token
        next_token, next_logprob, cache = step(token.reshape(1), cache)
        mx.async_eval(next_token, next_logprob)
    yield detok(token.item()), logprob  # the only read; step index + 1 is running
    if index + 1 < n:
        token, logprob = next_token, next_logprob
```

Do not call `mx.eval` anywhere, and never convert `token` to a Python `int` before building the next step.
