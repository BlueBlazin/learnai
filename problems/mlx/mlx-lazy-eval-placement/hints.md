After `logits, cache = model(...)`, `token = mx.argmax(...)`, and
`logprob = logprob + ...`, nothing has been computed: you hold three graphs. Which of
the things you hold will the next step read, and which will the caller read after the
loop? Those are the arrays that must have numbers behind them before the step ends.

---

Call `mx.eval` once at the end of each step and pass it everything that outlives the
step, in one call: `mx.eval(a, b, c)` evaluates a shared graph once, while three
separate calls schedule three graphs. Do not convert the token to a Python int to
feed it back: an evaluated scalar array can be reshaped to shape `(1,)` and passed to
the model as it is. Keep the tokens as arrays in the list you return.

---

The step, with its single evaluation:

```python
last = logits[-1]
logprobs = last - mx.logsumexp(last)
token = mx.argmax(logprobs)
logprob = logprob + logprobs[token]
mx.eval(token, logprob, cache)
```

Evaluating `token` alone would compute the cache as well, because the token is
computed *from* it. It would not compute `logprob`, which hangs off the side of the
graph. After the evaluation, feed `token.reshape(1)` with the cache, except after the
last token.
