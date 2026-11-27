## The idea

Every tensor lives on a single device, and an operation requires all of its operands to live on that same device. Operations on existing tensors preserve placement automatically, so multiplying two tensors produces a result on their shared device. Bugs come from tensors created without an existing parent tensor. A call like `torch.arange(T)` has no operand to inherit from, so it falls back to the default device (the CPU). To keep everything on one device, move external inputs once at the entry point and pass `device=` to every factory call (or use device-inheriting helpers like `torch.zeros_like(x)` and `x.new_zeros(...)`).

The second rule avoids host-device synchronization. Because GPU kernels run asynchronously, the CPU cannot read a device-computed value without waiting for all queued GPU work to finish and copying the value across. Calls such as `.item()`, `.tolist()`, `.cpu()`, `int(x)`, `if x:`, slicing with a tensor bound, and boolean-mask indexing (`h[mask]`, whose output size the host must learn) all trigger a synchronization. You can build a padding mask entirely on the device by broadcasting a row of position indices against a column of sequence lengths.

Dtypes follow the same principle. Combining a floating-point tensor with a `bool` or integer tensor preserves the float's dtype, so multiplying `h` by a boolean mask and dividing the sum by `lengths` stay in `h.dtype`. Combining two float tensors of different widths promotes to the wider type. Because calling `.float()` on a mask always produces `float32`, multiplying it with a `float16` tensor promotes the result to `float32`.

## Reference solution (PyTorch)

```python
def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(item, device) for item in batch)
    return batch


def run(weights, inputs, device):
    inputs = to_device(inputs, device)
    tokens, lengths = inputs["tokens"], inputs["lengths"]
    positions = torch.arange(tokens.shape[1], device=device)
    h = weights["tok_emb"][tokens] + weights["pos_emb"][positions]
    valid = positions[None, :] < lengths[:, None]
    pooled = (h * valid[:, :, None]).sum(dim=1) / lengths[:, None]
    return pooled @ weights["w_out"].T + inputs["head"]["bias"]
```

`tokens.shape[1]` is metadata stored on the host, so reading it does not synchronize with the device. Calling `tensor.to(device)` returns the original tensor with zero overhead when it is already on `device`.

## How the CPU cases see a GPU mistake

On a CPU run, every tensor is already on the CPU and readable without synchronization, so the test suite uses three checks. `DeviceSpy` wraps PyTorch factory functions and flags any call from your file that omits `device=`. `SyncSpy` wraps tensor methods that transfer values to Python and flags any call from your file. Finally, one test runs `run` on `torch.device("meta")`. Meta tensors carry shape and dtype without data, so elementwise operations reject mixing them with CPU tensors just as CUDA does. However, in PyTorch 2.15, indexing a meta tensor with a CPU index and matrix multiplication between meta and CPU tensors do not raise, and CUDA likewise allows indexing a CUDA tensor with a CPU index via an implicit transfer. The factory spy catches those cases too.

## Common mistakes

- **Calling `torch.arange(T)` without `device=`.** Position IDs are allocated on the CPU. Comparing them with GPU `lengths` raises an error, while using them to index a GPU embedding table silently transfers the index tensor.
- **Routing through NumPy, such as `torch.from_numpy(np.arange(T))`.** A tensor backed by a NumPy array lives in CPU memory, bypassing `device=` altogether.
- **Building the mask on the host.** Calling `lengths.cpu().tolist()` and constructing a mask in Python stalls the device and copies data in both directions.
- **Looping over rows with `h[b, :lengths[b]]`.** Using a tensor as a slice bound invokes `__index__`, forcing one host read per row.
- **Returning `.cpu()`.** The caller expects logits on `device` so downstream sampling or subsequent layers can run without another transfer.
- **Calling `.float()` on the mask or lengths.** `.float()` hard-codes `float32` and promotes a `float16` forward pass to `float32` from that point onward.
- **Using `h.mean(dim=1)`.** This averages over padding positions, making each row's output depend on the batch's padding length and pad tokens.
- **Writing a shallow `to_device`.** Moving only top-level dictionary values leaves `inputs["head"]["bias"]` on the CPU.
- **Turning tuples into lists, or mutating containers in place.** Callers that unpack tuples or reuse the original CPU batch for logging receive corrupted data.

## Complexity

The embedding lookup and masked sum take $O(B\,T\,D)$ time, and the linear head takes $O(B\,C\,D)$ time. Each input tensor is copied from host to device at most once, with zero device-to-host copies.

## In the wild

PyTorch's built-in modules follow these same conventions. For example, `nn.Transformer` creates masks on its input's device, and the PyTorch CUDA documentation recommends `new_*` and `*_like` constructors inside `forward`. Data-loader collate utilities and `Trainer` frameworks also rely on recursive `to_device` helpers. The CUDA behaviors described here follow the PyTorch 2.15 documentation on asynchronous execution and host-device synchronization. The CPU tests were executed during authoring, while the GPU tests were not.
