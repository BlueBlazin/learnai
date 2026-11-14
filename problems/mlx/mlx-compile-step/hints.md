`mx.compile(fn)` runs `fn` once, with placeholders in place of its array arguments,
records the operations, and afterwards replays the recording whenever the arguments
have the same shapes and dtypes. Go through everything your step reads. Which of
those are arguments (and so are fed fresh on every call), which are Python values
that never change, and which are arrays that the function reaches *without* their
being arguments?

---

Three separate things have to be right.

Shape and dtype stability: the state's shapes are fixed by `init_state`, but
`mx.argmax` returns uint32. If the first call receives an int32 token and returns a
uint32 one, the second call has a different input signature and is traced again.

The end of sequence: `if token == eos_id` asks for the *value* of a placeholder
while tracing, which MLX refuses. A choice that depends on values has to be an
operation in the graph: `mx.where(condition, a, b)`.

The parameters: `model.step` reads them from the module, not from the arguments. An
array that a compiled function reads that way is recorded as a constant unless you
declare it with `inputs=`.

---

```python
def step(token, state):
    logits, state = model.step(token, state)
    best = mx.argmax(logits, axis=-1).astype(mx.int32)
    return mx.where(token == eos_id, eos_id, best), state


return mx.compile(step, inputs=model.state)
```

Call `mx.compile` once, in `make_step`, and return its result: each call of
`mx.compile` makes a new function with an empty cache of traces. `inputs=model.state`
hands the compiled function the module's live dict, so the parameters are looked up
on every call; passing `model.parameters()` would hand it a copy of the tree made at
that moment.
