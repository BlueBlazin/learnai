`mx.async_eval(x)` returns immediately; the GPU starts computing `x` in the background,
and a later `x.item()` waits only for whatever is not finished yet. So what must
already be in the GPU's queue at the moment you block on token $t$? And what do you
need to know about token $t$ to *build* the graph of step $t + 1$: its value, or just
the array?

---

Keep two steps in hand: the current one (scheduled, about to be read) and the next
one (built from the current token **array** with `token.reshape(1)`, then scheduled).
Each step is `logits, cache = model(...)`, the log-softmax of `logits[-1]`, the argmax,
and the log-probability of the argmax. Pass both the token and its log-probability to
`mx.async_eval`: the log-probability is computed from the token, so scheduling the
token alone leaves it out. The cache needs no mention, because the token depends on
every array in it.

---

The loop:

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

Do not call `mx.eval` anywhere, and do not convert the token to an int before
building the next step.
