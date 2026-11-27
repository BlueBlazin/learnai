Break the problem into two questions. First, `mx.compile` keys its cache by the shapes of array arguments and the values of Python scalars. How can you map forty prompts of different lengths onto only three cache keys, and how can you pass the true prompt end position so that changing it does not create a new key? Second, `state["counts"]` is read and updated inside the function without being an argument. What happens during tracing when a function reads an external array or assigns to an external dict?

---

Do the padding in plain Python outside the compiled function. Right-pad the prompt of length `T` to the smallest bucket in `(16, 32, 64)` using `mx.pad(prompt, (0, bucket - T))`. Any padding token id works because the model is causal and you read logits at `T - 1`. Pass `T - 1` as a scalar array `last = mx.array(T - 1)` rather than a Python `int`, because Python scalars are part of the trace signature and trigger a new trace for each distinct value. Inside the compiled function, select row `T - 1` with `mx.take(logits, last, axis=0)`.

For `state["counts"]`, either declare it or pass it explicitly. With declaration, `mx.compile(fn, inputs=[model.state, state], outputs=state)` reads `state["counts"]` before every call and writes the updated array back into `state` afterward. With explicit threading, pass `state["counts"]` as an argument, return `(token, new_counts)`, and assign `state["counts"] = new_counts` in the Python wrapper.

---

```python
def pick(tokens, last):
    logits, _ = model(tokens)
    adjusted = mx.take(logits, last, axis=0) - penalty * state["counts"]
    token = mx.argmax(adjusted).astype(mx.int32)
    state["counts"] = state["counts"] + (mx.arange(adjusted.shape[0]) == token)
    return token


compiled = mx.compile(pick, inputs=[model.state, state], outputs=state)


def step(prompt):
    T = prompt.shape[0]
    bucket = next(size for size in (16, 32, 64) if size >= T)
    return compiled(mx.pad(prompt, (0, bucket - T)), mx.array(T - 1))
```

`mx.arange(adjusted.shape[0]) == token` builds a boolean one-hot vector over the vocabulary. Adding it to the `float32` counts array increments only the emitted token's entry by 1. Do not count nonzero tokens to determine `T`, because token `0` is a valid token.
