Implement a GPU batch prefetcher (`Prefetcher`) that overlaps host-to-device data transfers with model computation, and a result fetcher (`fetch_results`) that copies device outputs back to the host with at most one synchronization.

## Specification

```python
class Prefetcher:
    def __init__(self, batches, device, to_device, pin): ...
    def __iter__(self): ...
    def __next__(self): ...


def fetch_results(outputs, to_host, sync): ...
```

Normal CPU tensors live in **pageable** memory, which the operating system can move or swap to disk. To copy a tensor asynchronously in the background, the GPU driver requires **page-locked (pinned)** host memory.

To let the judge trace every call, the four PyTorch transfer primitives are passed in as arguments.

| argument | stands for | what it does |
|---|---|---|
| `pin(tensor)` | `tensor.pin_memory()` | Returns a new **copy** of a CPU tensor in page-locked (pinned) memory. The original tensor is unchanged and stays pageable. |
| `to_device(tensor, device, non_blocking=False)` | `tensor.to(device, non_blocking=...)` | Starts copying `tensor` to `device` and returns the resulting device tensor. When `non_blocking=True` and `tensor` is pinned, the call returns immediately while the copy runs in the background. Otherwise, the host waits for the copy to finish. |
| `to_host(tensor, non_blocking=False)` | `tensor.to("cpu", non_blocking=...)` | Copies a device tensor to host memory. A blocking call waits for the device and returns valid data. A non-blocking call returns immediately, but the returned tensor's **contents are undefined until the device is synchronized**. |
| `sync()` | `torch.cuda.synchronize()` | Waits until the device has finished all queued work, including pending transfers. |

**`Prefetcher(batches, device, to_device, pin)`** wraps `batches`, an iterable of CPU tensors (such as a list or a single-use generator). For a prefetcher instance `p`, `iter(p)` returns `p`. Each call to `next(p)` returns the next batch on `device` in the original order and raises `StopIteration` on every call after the last batch has been returned.

`Prefetcher` must follow three rules:

1. **Pinned, then non-blocking.** Pass each batch to `pin` once, then pass the returned pinned copy to `to_device` once with `non_blocking=True`. Return the device tensor produced by `to_device` from `next(p)`.
2. **One batch ahead.** The constructor pulls batch 0 from `batches` and starts its transfer. Before returning batch $i$, `next(p)` pulls batch $i + 1$ from `batches` (if one exists) and starts its transfer. Do not pull, pin, or transfer batch $i + 2$ until the following `next(p)` call.
3. **No buffer reuse.** Never overwrite a pinned copy after starting its transfer because the background copy may still be reading from it.

If `batches` is empty, pin and transfer nothing, and raise `StopIteration` on the first `next(p)` call.

![The copy of batch $i + 1$ starts when `next()` returns batch $i$, so it runs while the model works on batch $i$.](assets/prefetch-timeline.svg)

**`fetch_results(outputs, to_host, sync)`** takes a list of $n \ge 1$ device tensors with shapes $(b_i, \dots)$, matching trailing dimensions, and a common dtype. It returns a single host tensor formed by concatenating them in order along axis 0.

- The host may wait **at most once**. Each blocking `to_host` call counts as one wait, and each `sync()` call counts as one wait.
- Every output must pass through `to_host`, and you must not read from a non-blocking host tensor before the wait that makes its contents valid.

## Shapes

| name | shape | dtype |
|---|---|---|
| a batch in `batches` | any (may vary across batches) | any |
| a batch from `next` | same as its source batch | same as its source batch |
| `outputs[i]` | $(b_i, \dots)$ | one dtype for all |
| result of `fetch_results` | $(\sum_i b_i, \dots)$ | same dtype as `outputs` |

## Example

Suppose `batches` is a generator yielding three tensors. The judge records each time a batch is pulled from the source, pinned, transferred, and returned by `next`:

```python
p = Prefetcher(batches, device, to_device, pin)
for batch in p:
    ...  # the model runs on `batch`
```

| during | events, in order |
|---|---|
| constructor | take 0, pin 0, transfer 0 |
| first `next` | take 1, pin 1, transfer 1, then return batch 0 |
| second `next` | take 2, pin 2, transfer 2, then return batch 1 |
| third `next` | source is exhausted, then return batch 2 |
| fourth `next` | `StopIteration` |

While the model runs on batch 0, batch 1 is already copying in the background and batch 2 has not been pulled yet. A prefetcher that transfers on demand produces the same values, but in the order take 0, pin 0, transfer 0, return 0, take 1, and so on, leaving the model idle during every copy.

For `fetch_results`, suppose `outputs` contains three tensors with 2, 4, and 1 rows:

```python
staged = [to_host(out, non_blocking=True) for out in outputs]  # returns immediately, three times
# staged[0] holds undefined data here
sync()  # the single host wait
# all three tensors now hold valid data
```

Concatenating `staged` along axis 0 after `sync()` produces a host tensor with 7 rows. Reading `staged` before `sync()` reads undefined memory. In CPU tests, the judge's `to_host` fills unsynchronized memory with `NaN` for floats and `-1` for integers until `sync()` is called.

## Constraints

- `batches` contains 0 to 16 tensors. `fetch_results` receives 1 to 16 tensors.
- Integer tensors are compared exactly, and floating-point tensors are compared with a relative tolerance of `1e-4` (since no math is performed, a correct solution matches exactly). Each batch is verified when returned and again after the loop finishes.
- The pin count, transfer count, `non_blocking` flag, and event ordering must match the specification exactly.
- `fetch_results` may perform at most 1 host wait, reported as `host_waits`.
- **Execution environment.** Standard test cases run on the CPU using recording stand-ins for the four functions. In those tests, "device" tensors share memory with their pinned copies, so overwriting a pinned buffer too early corrupts a batch that was already handed out. Two additional cases run only on machines with an NVIDIA GPU using `torch.Tensor.pin_memory`, real `.to(...)` transfers, and `torch.cuda.synchronize`. Those GPU cases verify that each transferred tensor arrives on CUDA with the right values from a tensor where `is_pinned()` is true, and that fetched results are complete. They were written from the PyTorch 2.15 documentation and could not be executed when the problem was authored. No GPU case measures wall-clock time.

## Why this matters

A GPU decode step takes milliseconds, and copying token IDs or image features across PCIe takes comparable time. A loop that copies, computes, and copies back sequentially leaves the GPU idle during every transfer and the CPU idle during every kernel.

Overlapping transfers with compute requires two conditions at once. The source tensor must be page-locked so `non_blocking=True` actually runs asynchronously, and the next transfer must start before the current step begins. In the reverse direction, skipping synchronization is worse than a slowdown. Reading a host tensor before the device finishes writing it returns stale or garbage bytes, a race condition that often hides in small tests whenever the GPU finishes quickly.
