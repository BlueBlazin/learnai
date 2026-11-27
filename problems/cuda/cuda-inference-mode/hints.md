`model.eval()` changes what dropout and batch normalization compute, but it does not stop autograd from recording. `torch.inference_mode()` stops autograd recording, but it does not change what any layer computes. Which of the two can you restore with a context manager, and which one must you restore yourself?

---

For `predict`, save the `training` flag of every module in `model.modules()` before modifying the model. Saving only the root flag fails because `model.train(flag)` overwrites every submodule with one value. Call `model.eval()`, move `x` with `x.to(device)`, run the forward pass inside `with torch.inference_mode():`, and restore the saved flags inside a `finally` block so they are restored even if `model` raises.

For `as_inference`, calling a generator function does not execute its body. It only creates the generator object. A wrapper like `with mode: return fn(...)` exits the mode before the first step runs, while `with mode: yield from fn(...)` keeps the mode active while the caller runs between steps.

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

`torch.inference_mode()` also works as a decorator and handles generator functions automatically. It enters inference mode each time the generator resumes and exits at each `yield`, so `as_inference(fn)` can simply return `torch.inference_mode()(fn)`.
