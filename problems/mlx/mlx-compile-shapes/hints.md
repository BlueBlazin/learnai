Two separate questions. First: the compiled function is keyed by the shapes of its
array arguments, so what must be true of the token array you hand it for forty
prompts to need only three keys, and how does the function then know where the
prompt ends? (What kind of value can change from call to call *without* changing the
key?) Second: `state["counts"]` is read and assigned inside the function but is not
an argument. What does a trace record for an array it reads that way, and when does
an assignment inside a traced function actually run?

---

Pad in plain Python, outside the compiled function: extend the prompt on the right
to 16, 32, or 64 tokens (`mx.pad(prompt, (0, bucket - T))`; any token id will do as
padding, since the model is causal and you read the logits at position `T - 1`).
Pass that position as an **array**, `mx.array(T - 1)`: a Python int argument is part
of the compiled function's signature, and each new value would be a new trace. Inside,
select the row with `mx.take(logits, last, axis=0)`.

For the state, either declare it or make it explicit. Declared:
`mx.compile(fn, inputs=state, outputs=state)` reads `state["counts"]` from the dict at
every call and writes the array assigned inside `fn` back into the dict after every
call. Explicit: pass `state["counts"]` as an argument, return the new counts, and
assign them to the dict in the Python wrapper.

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

`mx.arange(V) == token` is a one-hot row of booleans; adding it to the float32 counts
adds 1 at the token's index. Do not count nonzero tokens to find the length: token 0
is a real token.
