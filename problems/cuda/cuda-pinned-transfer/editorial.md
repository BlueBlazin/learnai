## The idea

**Host to device.** The GPU driver can only copy from page-locked host memory,
memory the operating system has promised not to move or swap out. An ordinary
(pageable) tensor therefore has to be staged into a page-locked buffer first, and the
host is busy doing that. PyTorch's documentation of `Tensor.cuda` states the
consequence: with `non_blocking=True` the copy is asynchronous with respect to the
host when the source is in pinned memory, and otherwise "the argument has no effect".
Pinning is the precondition; `non_blocking=True` on a pageable tensor is accepted,
does the copy, and overlaps with nothing.

`tensor.pin_memory()` is not an in-place operation. It copies the tensor to pinned
memory (unless it is pinned already) and returns the copy; the tensor it was called
on is as pageable as before.

An asynchronous copy only helps if something else happens meanwhile. So the copy of
batch $i + 1$ must be *started* before the model starts on batch $i$: that is the
look-ahead. One batch is enough, because the copy is much shorter than the model's
work, and more than one costs device memory for batches that are not needed yet
(with the whole source transferred up front, the dataset has to fit on the GPU).

No synchronization is needed on this side. Operations on a device run in the order
they were queued, so a kernel that uses a batch runs after the copy that fills it.
What can go wrong is on the host: if the pinned source is overwritten while the copy
is still reading it, the device receives a mixture. PyTorch's tutorial on the subject
demonstrates exactly this with a pinned tensor that is zeroed right after a
non-blocking transfer. Hence one pinned copy per batch, left alone.

**Device to host.** Here the order guarantee does not reach the reader. A
non-blocking `to("cpu")` returns a host tensor immediately, and the host can read
host memory whenever it likes, including before the device has written to it.
`torch.cuda.synchronize()` waits for everything queued, after which the data is
there. The cost model is simple: each blocking copy is a wait, each synchronize is a
wait, and one wait after all copies have been queued is enough for all of them.

## Reference solution (PyTorch)

```python
class Prefetcher:
    def __init__(self, batches, device, to_device, pin):
        self._source = iter(batches)
        self._device, self._to_device, self._pin = device, to_device, pin
        self._in_flight = None
        self._done = False
        self._start_next()

    def _start_next(self):
        try:
            batch = next(self._source)
        except StopIteration:
            self._in_flight, self._done = None, True
            return
        pinned = self._pin(batch)
        self._in_flight = self._to_device(pinned, self._device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        current = self._in_flight
        self._start_next()
        return current


def fetch_results(outputs, to_host, sync):
    staged = [to_host(out, non_blocking=True) for out in outputs]
    sync()
    return torch.cat(staged, dim=0)
```

`_done` becomes true when a call of `_start_next` finds the source empty. In
`__next__` that happens *after* the current batch has been set aside, so the last
batch is returned and the call after it raises. For an empty source the constructor
already sets it. `iter(batches)` makes a list and a generator behave alike and
guarantees that nothing is taken from the source twice.

`fetch_results` could also concatenate on the device and make one blocking copy; that
is one wait too, at the price of a second copy of the results in device memory.

## Common mistakes

- **`non_blocking=True` from pageable memory.** No error, no warning, no overlap.
- **Throwing away the result of `pin`**: `batch.pin_memory()` on its own line, then
  `batch.to(device, non_blocking=True)`. The tensor that is transferred is the
  pageable one.
- **Pinning and then transferring without `non_blocking=True`.** The host waits for
  every copy.
- **Handing out the pinned host tensor** instead of what `to_device` returned. On a
  CPU run nothing fails; on a GPU the model receives a CPU tensor.
- **Transferring on demand**, or **starting the first transfer at the first `next`**.
  The values and the pinning are right and each copy happens while the caller waits
  for it.
- **Dropping the last batch.** The loop "while there is a next batch, return the
  current one" stops one batch early: when the look-ahead finds nothing, a batch is
  still in flight.
- **Transferring everything in the constructor.** Everything overlaps, and the device
  holds the whole source.
- **Reusing one pinned buffer.** Writing batch $i + 1$ into the buffer while batch
  $i$'s copy may still be reading it. With a look-ahead it takes two buffers used in
  turn, and proof that the older copy has finished, to do this correctly.
- **Reading before the wait** in `fetch_results` (`torch.cat` before `sync()`, or a
  `sync()` before the copies are queued, which waits for the model and not for the
  copies).
- **A blocking copy per output**, or **a `sync()` after each non-blocking copy**. The
  data is right and the host stalls once per output.
- **Not transferring at all**: on a CPU run `torch.cat(outputs)` is already a host
  tensor; on a GPU it is not.

## Complexity

One pinned copy and one transfer per batch; at most two batches on the device on
behalf of the prefetcher (the one handed out and the one in flight). `fetch_results`
makes one copy per output, one wait, and one concatenation.

## In the wild

`DataLoader(pin_memory=True)` does the pinning in a separate thread, so that the
page-locked copy is not paid for by the thread that feeds the model; the training
loop then calls `.to(device, non_blocking=True)` on batches that arrive pinned. That
detail matters: PyTorch's tutorial "A guide on good usage of non_blocking and
pin_memory()" measures that calling `pin_memory()` on the main thread immediately
before the transfer is usually no faster than a plain `.to(device)`, because the
blocking copy into page-locked memory is the same work the driver would have done.
In this problem `pin` is an argument for that reason: where the pinning happens is
the caller's business, and the prefetcher's job is the order of operations. The
classic "data prefetcher" of GPU training scripts is this class with a side stream
for the copies.

What was verified where: on the CPU build of torch 2.15, `pin_memory()` returns a
new tensor for which `is_pinned()` is true and leaves its argument unpinned. The
statements about asynchronous copies, about undefined data after a non-blocking copy
to the host, and about overwriting a pinned source are taken from the 2.15
documentation (`Tensor.cuda`, the CUDA notes, and the tutorial named above); the GPU
cases have not been run.
