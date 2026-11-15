For the prefetcher: at the moment `next` is about to return batch 3, which copy
should already be running so that it overlaps with the model's work on batch 3? Who
has to start it, and when is the only moment before the first `next` at which
anything can be started? For the results: after `to_host(x, non_blocking=True)`
returns, what is in the tensor it gave you, and what has to happen before you may
look?

---

Keep one piece of state: the batch whose transfer is under way. A helper takes the
next batch from `iter(batches)`, pins it, starts the transfer of the *pinned copy*
with `non_blocking=True`, and stores the tensor that `to_device` returned; when the
source is exhausted it records that instead. Call the helper once in the constructor.
In `__next__`, set aside the stored tensor, call the helper again, and return what
you set aside. `pin` returns a new tensor and leaves its argument alone, so use the
return value, and call `pin` for every batch: a pinned buffer that is still being
copied from must not be overwritten.

For `fetch_results`, start all the copies first, wait once, and only then read.

---

```python
def __next__(self):
    if self._done:
        raise StopIteration
    current = self._in_flight
    self._start_next()  # pins and transfers the following batch, or sets self._done
    return current
```

Setting `self._done` only when the source is found empty *during* this call means the
last batch, which was already in flight, is still returned. And

```python
staged = [to_host(out, non_blocking=True) for out in outputs]
sync()
return torch.cat(staged, dim=0)
```
