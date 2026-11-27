## The idea

**Host to device.** The GPU driver can only run asynchronous DMA (Direct Memory Access) copies from page-locked host memory that the operating system will not move or swap out. When you transfer an ordinary pageable tensor, the driver first copies it into a temporary page-locked staging buffer on the host, keeping the CPU busy. As PyTorch's `Tensor.cuda` documentation notes, `non_blocking=True` makes a copy asynchronous with respect to the host only when the source is pinned. On a pageable tensor, the argument is accepted silently and has no effect.

`tensor.pin_memory()` is not an in-place operation. Unless the tensor is already pinned, it returns a new pinned copy and leaves the original tensor pageable.

An asynchronous copy only saves time if useful work runs alongside it. Starting the copy of batch $i + 1$ before returning batch $i$ gives a one-batch look-ahead, so the transfer overlaps with the model's forward and backward pass on batch $i$. One batch ahead is usually enough because the transfer is shorter than the model step, while prefetching many batches wastes GPU memory (prefetching everything up front requires the entire dataset to fit on the device).

No explicit synchronization is needed in this direction. Operations queued on the same CUDA stream execute in order, so any kernel that consumes a batch waits until its copy completes. The hazard is on the host side. If you overwrite a pinned buffer while the background copy is still reading from it, the device receives corrupted data. PyTorch's tutorial on `non_blocking` and `pin_memory` demonstrates this by zeroing a pinned tensor immediately after launching a non-blocking transfer. Allocating a fresh pinned copy per batch avoids this race.

**Device to host.** Stream ordering does not protect the CPU when reading host memory. A non-blocking `to("cpu")` returns a host tensor right away, and the CPU can read from it before the GPU has written the data. Calling `torch.cuda.synchronize()` waits for all queued device work to finish so every staged host tensor is safe to read. Each blocking copy and each `sync()` costs one host wait, so queuing all non-blocking copies first and synchronizing once at the end minimizes stalls.

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

`_done` becomes `True` only when `_start_next` finds the source exhausted. In `__next__`, that happens *after* `current` has been saved, so the final in-flight batch is still returned and the next call raises `StopIteration`. When `batches` is empty, the constructor sets `_done = True` right away. Calling `iter(batches)` once lets lists and generators behave identically and ensures each batch is pulled only once.

An alternative for `fetch_results` is to concatenate `outputs` on the device and run one blocking `to_host` call. That also uses one wait, at the cost of allocating an extra concatenated copy in GPU memory.

## Common mistakes

- **Passing `non_blocking=True` on pageable memory.** PyTorch raises no error or warning, but the transfer blocks and overlaps with nothing.
- **Ignoring the return value of `pin`.** Calling `batch.pin_memory()` on its own line and then transferring `batch` still copies the original pageable tensor.
- **Pinning without `non_blocking=True`.** The host still blocks on every transfer.
- **Returning the pinned host tensor** instead of the tensor returned by `to_device`. This passes silently on CPU tests but hands the model a CPU tensor on a GPU.
- **Transferring on demand or starting the first transfer in `__next__`.** The values and pinning are valid, but the caller waits for every copy instead of overlapping it.
- **Dropping the last batch.** Stopping as soon as the look-ahead hits `StopIteration` discards the batch that was already in flight.
- **Transferring the entire source in `__init__`.** Every copy starts early, but the GPU must hold the whole dataset at once.
- **Reusing one pinned buffer.** Writing batch $i + 1$ into a buffer while batch $i$'s copy may still read from it corrupts batch $i$. Safe buffer reuse requires double-buffering and explicit proof that the earlier copy finished.
- **Reading before waiting in `fetch_results`.** Calling `torch.cat` before `sync()`, or calling `sync()` before queuing the copies, reads undefined memory.
- **Waiting once per output.** Using blocking `to_host` calls or calling `sync()` inside the loop produces valid data but stalls the host on every tensor.
- **Skipping `to_host` completely.** On CPU tests, `torch.cat(outputs)` already yields a CPU tensor, but on a GPU it stays on the device.

## Complexity

`Prefetcher` performs one pin and one transfer per batch and holds at most two batches on the device at a time (the returned batch and the one in flight). `fetch_results` performs one transfer per output, one synchronization, and one concatenation.

## In the wild

PyTorch's `DataLoader(pin_memory=True)` pins batches in a background worker thread so the main training loop only calls `.to(device, non_blocking=True)`. That separation matters. PyTorch's tutorial *A guide on good usage of non_blocking and pin_memory()* shows that calling `pin_memory()` on the main thread right before `.to()` is rarely faster than a plain blocking `.to(device)` because the main thread still pays for the copy into page-locked memory. Here `pin` is passed in as an argument so the caller controls where pinning happens while `Prefetcher` controls the transfer ordering. Production GPU training scripts use this same prefetcher pattern, often pairing it with a dedicated CUDA stream for copies.

Verification note: on the CPU build of PyTorch 2.15, `pin_memory()` returns a new tensor with `is_pinned() == True` and leaves its input unpinned. The behavior of asynchronous copies, undefined data before `sync()`, and overwriting pinned buffers is taken from the PyTorch 2.15 documentation (`Tensor.cuda`, CUDA semantics notes, and the tutorial above). The GPU test cases have not been run.
