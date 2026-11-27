A graph replay runs no Python and takes no arguments. Think of the recorded graph as a fixed sequence of commands that read from address A and write to address B. From outside the graph, what are the only two memory operations you can perform to supply a new input before replay and preserve an output after replay?

---

In the constructor, first allocate placeholders outside capture with `self.static_inputs = tuple(x.clone() for x in example_inputs)`. Next, run `step_fn(*self.static_inputs)` `warmup` times inside `with side_stream(device):`. Finally, capture once with nothing else in the block:

```python
with torch.cuda.graph(graph):
    self.static_outputs = step_fn(*self.static_inputs)
```

Store `graph`, `self.static_inputs`, and `self.static_outputs` on `self`. You do not need to keep `step_fn` after the constructor finishes.

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

Check shapes yourself before copying. Otherwise, `copy_` silently broadcasts a `(1, n)` input across every row of an `(m, n)` placeholder.
