A replay runs no Python and takes no arguments. So how can it compute something
different each time, and where does its result go? Think of the graph as a list of
"run this kernel on address A, write to address B": what are the only two things you
can do from outside to give it a new input and to keep an old output?

---

In the constructor: `self.static_inputs = tuple(x.clone() for x in example_inputs)`
(before the capture, never inside it), then `warmup` plain calls
`step_fn(*self.static_inputs)` inside `with side_stream(device):`, then

```python
with torch.cuda.graph(graph):
    self.static_outputs = step_fn(*self.static_inputs)
```

with nothing else in the block. Keep `graph`, the static inputs, and the static
outputs on `self`; you do not need `step_fn` after the constructor.

---

```python
def __call__(self, *inputs):
    # ... raise ValueError on a wrong count or shape first ...
    for static, new in zip(self.static_inputs, inputs, strict=True):
        static.copy_(new)  # into the captured memory; never `static = new`
    self.graph.replay()
    if isinstance(self.static_outputs, torch.Tensor):
        return self.static_outputs.clone()  # after the replay, not before
    return tuple(out.clone() for out in self.static_outputs)
```

Compare shapes yourself: `copy_` broadcasts, so a `(1, n)` input would silently fill
every row of an `(m, n)` placeholder.
