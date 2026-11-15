At the moment your `except` block starts, what still refers to the tensors of the
call that failed? Think of the exception object, what an exception carries with it
so that a traceback can be printed, and your own local variables. Which of these are
gone once the `except` block has ended?

---

Write a `while start < N` loop with three variables: `start`, `size`, and a count of
consecutive successes. Inside the `try`, call the model on the slice without binding
the slice to a name. In the `except torch.OutOfMemoryError` block do only two things:
re-raise if the batch had one row, otherwise set a flag. After the `try` statement,
if the flag is set, call `empty_cache()`, set `size` to half of the number of rows
that failed, reset the count, and `continue` without advancing. Otherwise advance,
and handle `grow_after`. Catch `torch.OutOfMemoryError`, not `RuntimeError`.

---

```python
end = min(start + size, n)
oom = False
try:
    outputs.append(model(items[start:end]))
except torch.OutOfMemoryError:
    if end - start == 1:
        raise
    oom = True
if oom:
    empty_cache()
    size = (end - start) // 2
    streak = 0
    continue
```

After a success: `start = end`; then, if `grow_after is not None and size <
batch_size`, add one to `streak`, and when it reaches `grow_after` set
`size = min(2 * size, batch_size)` and `streak = 0`.
