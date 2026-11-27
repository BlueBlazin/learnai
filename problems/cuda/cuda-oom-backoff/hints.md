When your `except` block starts, what still holds references to the tensors from the failed call? Consider the exception object, the traceback it carries, and your own local variables. Which of those references disappear once the `except` block finishes?

---

Use a `while start < N` loop tracking `start`, `size`, and a streak of consecutive successes. Inside `try`, pass `items[start:end]` directly to `model` without assigning it to a local variable. Inside `except torch.OutOfMemoryError:`, only re-raise if the batch had 1 row or set a flag otherwise. After the `try`/`except` block, if the flag is set, call `empty_cache()`, halve the number of rows that failed, reset the streak, and `continue` without advancing `start`. Otherwise advance `start` and update `grow_after`. Make sure to catch `torch.OutOfMemoryError`, not `RuntimeError`.

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

After a successful batch, set `start = end`. If `grow_after is not None and size < batch_size`, increment `streak`, and when `streak == grow_after` set `size = min(2 * size, batch_size)` and `streak = 0`.
