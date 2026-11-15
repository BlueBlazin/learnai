`model.eval()` changes what dropout and batch normalization compute; it does not stop
autograd from recording. `torch.inference_mode()` stops the recording; it does not
change what any layer computes. Which of the two can you undo with a context manager,
and which one do you have to undo yourself?

---

For `predict`: before touching the model, save the `training` flag of every module in
`model.modules()`, not only the root's, because `model.train(flag)` writes one value
into all of them. Call `model.eval()`, run the forward pass inside
`with torch.inference_mode():`, and put the flags back in a `finally` block so that
they are restored when the model raises. Move `x` with `x.to(device)`.

For `as_inference`: calling a generator function runs none of its body, it only
creates the generator. A wrapper of the form `with mode: return fn(...)` therefore
leaves the mode before the first step runs, and `with mode: yield from fn(...)` keeps
it on while the caller's code runs between steps.

---

```python
flags = [(m, m.training) for m in model.modules()]
model.eval()
try:
    with torch.inference_mode():
        return model(x.to(device))
finally:
    for m, training in flags:
        m.training = training
```

`torch.inference_mode()` is also a decorator, and as a decorator it handles generator
functions correctly: it enters the mode each time the generator resumes and leaves it
at each `yield`. So `as_inference(fn)` can be `torch.inference_mode()(fn)`.
