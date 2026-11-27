Think about timing in both directions. When `next` is about to return batch 3, which batch's copy should already be running so it overlaps with the model's work on batch 3? And when is the only moment before the first `next` call to start batch 0?

For `fetch_results`, what does `to_host(x, non_blocking=True)` put in the returned tensor immediately, and what must happen before you read it?

---

Store one piece of state in `Prefetcher`: the batch whose transfer is currently in flight. Write a helper that pulls the next batch from `iter(batches)`, pins it, starts transferring the *pinned copy* with `non_blocking=True`, and stores the tensor returned by `to_device`. If the source is exhausted, record that the iterator is done instead.

Call that helper once in `__init__`. In `__next__`, save the current in-flight tensor, call the helper again to start the next transfer, and return the saved tensor. Always use the return value of `pin` (it does not modify its input) and call `pin` fresh for each batch so you never overwrite a buffer that is still being copied.

In `fetch_results`, launch all non-blocking `to_host` copies first, call `sync()` once, and only then concatenate.

---

```python
def __next__(self):
    if self._done:
        raise StopIteration
    current = self._in_flight
    self._start_next()  # pins and transfers the following batch, or sets self._done
    return current
```

Setting `self._done` only when `_start_next` finds the source empty ensures the last in-flight batch is still returned before `StopIteration` is raised on the next call.

```python
staged = [to_host(out, non_blocking=True) for out in outputs]
sync()
return torch.cat(staged, dim=0)
```
