After `logits, cache = model(...)`, `token = mx.argmax(...)`, and `logprob = logprob + ...`, nothing has been computed yet, so you hold three unevaluated graphs. Ask which arrays the next step will read and which ones the caller will read after the loop. Those are the arrays that must hold concrete values before the step ends.

---

Call `mx.eval` once at the end of each step and pass everything that outlives the step in a single call. `mx.eval(a, b, c)` evaluates a shared graph once, whereas three separate calls schedule three graphs. Do not convert `token` to a Python `int` to feed it back into the model. An evaluated scalar array can be reshaped to `(1,)` directly, and the returned `tokens` list expects MLX arrays.

---

Here is the step body with its single evaluation:

```python
last = logits[-1]
logprobs = last - mx.logsumexp(last)
token = mx.argmax(logprobs)
logprob = logprob + logprobs[token]
mx.eval(token, logprob, cache)
```

Evaluating `token` alone also computes `cache` because `token` depends on it, but it does not compute `logprob`, which sits off to the side of the graph. After `mx.eval`, feed `token.reshape(1)` and `cache` back into `model`, except after the last token.
