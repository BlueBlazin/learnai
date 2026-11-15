Feed batches to a model on a GPU so that the copy of the next batch runs while the
model works on the current one, and bring the results back with a single wait. Both
directions use `non_blocking=True`, and both have a way to do so that looks right and
is not: a host-to-device copy that silently blocks, and a device-to-host copy that is
read before it has arrived.

## Specification

```python
class Prefetcher:
    def __init__(self, batches, device, to_device, pin): ...
    def __iter__(self): ...
    def __next__(self): ...


def fetch_results(outputs, to_host, sync): ...
```

The functions a solution would take from PyTorch are passed in, so that the judge can
watch them:

| argument | stands for | what it does |
|---|---|---|
| `pin(tensor)` | `tensor.pin_memory()` | Returns a **copy** of a CPU tensor in page-locked (pinned) memory. The argument itself is not changed and stays pageable. |
| `to_device(tensor, device, non_blocking=False)` | `tensor.to(device, non_blocking=...)` | Starts the copy and returns the tensor on `device`. With `non_blocking=True` and a pinned source it returns at once and the copy proceeds in the background; otherwise the host waits for the copy. |
| `to_host(tensor, non_blocking=False)` | `tensor.to("cpu", non_blocking=...)` | Copies a device tensor to host memory. A blocking call waits for the device and returns the data. A non-blocking call returns at once a tensor whose **contents are undefined until the device has been synchronized**. |
| `sync()` | `torch.cuda.synchronize()` | Waits until the device has finished everything queued, copies included. |

**`Prefetcher(batches, device, to_device, pin)`** is an iterator over `batches`, an
iterable of CPU tensors (a list, or a generator that can be consumed only once).
`iter(p)` returns `p`; `next(p)` returns the batches in the order of the source, each
on `device`, and raises `StopIteration` after the last one, however often it is called
again.

1. **Pinned, then non-blocking.** Each batch is passed to `pin` exactly once, and the
   pinned copy is passed to `to_device` exactly once, with `non_blocking=True`. What
   `next` returns is the tensor that `to_device` returned.
2. **One batch ahead.** The constructor takes the first batch from the source and
   starts its transfer. Each call of `next`, before it returns batch $i$, takes batch
   $i + 1$ from the source (if there is one) and starts its transfer. Batch $i + 2$ is
   not taken from the source, pinned, or transferred until `next` is called again.
3. **Buffers are not reused.** A pinned copy must not be written to after its transfer
   has started: the copy may still be reading from it.

An empty source is valid: nothing is pinned or transferred, and the first `next`
raises `StopIteration`.

![The copy of batch $i + 1$ starts at the moment `next()` returns batch $i$, so it runs while the model is busy with batch $i$. The first copy is started by the constructor.](assets/prefetch-timeline.svg)

**`fetch_results(outputs, to_host, sync)`** takes a list of $n \ge 1$ tensors on a
device, of shapes $(b_i, \dots)$ with equal trailing dimensions and one dtype, and
returns a single host tensor: their concatenation along axis 0, in order.

- The host may wait **at most once**. Every blocking `to_host` call counts as one
  wait, and so does every `sync()`.
- The data must have gone through `to_host`, and what is returned must be the data:
  nothing read from a host tensor before the wait that makes it valid.

## Shapes

| name | shape | dtype |
|---|---|---|
| a batch in `batches` | any, may differ between batches | any |
| a batch from `next` | the shape of its source | the dtype of its source |
| `outputs[i]` | $(b_i, \dots)$ | one dtype for all |
| result of `fetch_results` | $(\sum_i b_i, \dots)$ | the same dtype |

## Example

Three batches, taken from a generator. The judge records each time a batch is taken
from the source, pinned, transferred, and handed out by `next`:

```python
p = Prefetcher(batches, device, to_device, pin)
for batch in p:
    ...  # the model runs on `batch`
```

| during | events, in order |
|---|---|
| the constructor | take 0, pin 0, transfer 0 |
| the first `next` | take 1, pin 1, transfer 1; then batch 0 is handed out |
| the second `next` | take 2, pin 2, transfer 2; then batch 1 is handed out |
| the third `next` | the source is empty; batch 2 is handed out |
| the fourth `next` | `StopIteration` |

While the model runs on batch 0, the copy of batch 1 is under way, and batch 2 is
still in the source. A prefetcher that transfers each batch only when it is asked for
has the same values and this order instead: take 0, pin 0, transfer 0, hand out 0,
take 1, ... with the model idle during every copy.

For `fetch_results`, with three outputs of 2, 4, and 1 rows:

```python
staged = [to_host(out, non_blocking=True) for out in outputs]  # returns at once, three times
# staged[0] holds undefined data here
sync()  # the one wait
# now all three hold the data
```

The result has 7 rows. Concatenating before the `sync()` builds it from the undefined
contents; the judge's `to_host` fills such memory with NaN (with $-1$ for integers)
until `sync()` is called.

## Constraints

- A source holds 0 to 16 batches; `fetch_results` gets 1 to 16 outputs.
- Batches and results are compared exactly for integers and to a relative tolerance
  of `1e-4` for floats (nothing is computed, so a correct solution matches exactly).
  Each batch is checked when it is handed out and again after the loop has finished.
- The pin count, the transfer count, the `non_blocking` flag, and the order rules of
  the specification are exact.
- `fetch_results`: at most 1 host wait, reported as `host_waits`.
- **What runs where.** The cases without a hardware mark run on the CPU, where the
  judge supplies recording stand-ins for the four functions; its "device" tensors
  share memory with the pinned copies, so a pinned buffer that is overwritten too
  early changes a batch that was already handed out. Two cases run only on a machine
  with an NVIDIA GPU, with `torch.Tensor.pin_memory`, real `.to(...)` copies, and
  `torch.cuda.synchronize`: the batches must arrive on the CUDA device with the right
  values from tensors for which `is_pinned()` is true, and the fetched results must
  be complete. The GPU cases were written from the PyTorch 2.15 documentation and
  could not be run when the problem was authored; no GPU case measures time.

## Why this matters

A decode step on a GPU takes milliseconds, and so does copying a batch of token ids
or image features across PCIe. A serving loop that copies, then computes, then copies
back leaves the GPU idle during every transfer and the host idle during every
computation. Overlapping them needs two things at once: the source of an
asynchronous copy has to be page-locked, or `non_blocking=True` has no effect, and
the copy has to be *started early*, which is what the look-ahead is for. In the other
direction the failure is worse than slowness: results read from the host before the
device has delivered them are whatever the memory held, and code that does this
passes its tests whenever the GPU happens to be fast enough.
