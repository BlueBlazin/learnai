## The idea

A tensor lives on one device, and an operation needs all its operands on the same
one. Operations on existing tensors take care of themselves: the result of
`h * valid` is created where `h` and `valid` are. The tensors that go wrong are the
ones with no parent: those made by a factory function. `torch.arange(T)` has no
operand to inherit a device from, so it uses the default device, which is the CPU
unless someone changed it. The rule that keeps a forward pass on one device is
therefore short: **every factory call says `device=`** (or uses a form that inherits,
such as `torch.zeros_like(x)` and `x.new_zeros(...)`), and every tensor that arrives
from outside is moved once, at the door.

The second half of the rule is about values, not placement. The host cannot know a
number that the device has computed without waiting for the device to finish
everything queued before it and copying the number over. `.item()`, `.tolist()`,
`.cpu()`, `int(x)`, `if x:`, a tensor as a slice bound, and an index with a boolean
mask (whose result has a size the host must learn) all do that. A padding mask does
not need the lengths as Python numbers: comparing a row of positions with a column of
lengths says the same thing, entirely on the device.

Dtypes follow the same logic as devices. `bool` and integer tensors do not widen a
float tensor they are combined with, so `h * valid` and `sum / lengths` stay in the
dtype of `h`. A float tensor of another width does: `valid.float()` is float32 by
definition, and float16 times float32 is float32.

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

`tokens.shape[1]` is metadata kept on the host; reading it synchronizes nothing.
`tensor.to(device)` returns the tensor itself when it is already there, so moving a
batch that is already on the device costs nothing.

## How the CPU cases see a GPU mistake

On a CPU run every tensor is on the CPU and every value is already there, so the
tests use three instruments. `DeviceSpy` wraps the factory functions and reports each
call from your file that had no `device=`. `SyncSpy` wraps the tensor methods that
hand a value to Python and reports each call from your file. And one case runs `run`
on `torch.device("meta")`: meta tensors carry a shape and a dtype and no data, and
elementwise operations refuse to mix them with CPU tensors with the same kind of
error CUDA gives. The meta device is not a complete stand-in: indexing a meta tensor
with a CPU index tensor and `@` between the two do not raise in PyTorch 2.15, and
indexing a CUDA tensor with a CPU index is also accepted on a GPU, where it means a
transfer. That is why the factory check exists as well.

## Common mistakes

- **`torch.arange(T)` without `device=`.** The position ids are on the CPU. The
  comparison with the lengths raises on a GPU; the embedding lookup with a CPU index
  does not, and silently ships the index across.
- **`torch.from_numpy(np.arange(T))`** and other detours through NumPy. A tensor
  wrapped around host memory is a CPU tensor, with no factory call to warn about it.
- **Building the mask on the host.** `lengths.cpu().tolist()` and a list
  comprehension give the right mask, after waiting for the device and copying in both
  directions.
- **A Python loop over the rows** with `h[b, :lengths[b]]`. A tensor used as a slice
  bound is converted with `__index__`, a host read per row.
- **Returning `.cpu()`.** The caller asked for logits on the device; sampling and the
  next layer run there.
- **`.float()` on the mask or the lengths.** It hard-codes float32 and promotes a
  half-precision pass to single precision from that line on.
- **`h.mean(dim=1)`.** Padding positions count, and each row's result depends on what
  the batch happened to be padded with and to.
- **A shallow `to_device`.** Moving the values of the top-level dict leaves
  `inputs["head"]["bias"]` on the CPU.
- **Turning tuples into lists**, or **moving in place**: the caller that unpacks a
  pair, or reuses the CPU batch for logging, gets something it did not send.

## Complexity

$O(B\,T\,D)$ for the embeddings and the pooled sum, $O(B\,C\,D)$ for the head; one
host-to-device copy per input tensor, and no copy in the other direction.

## In the wild

PyTorch's own modules follow this rule (masks in `nn.Transformer` are created with
the device of their input), and the CUDA notes recommend the `new_*` and `*_like`
constructors inside `forward` for the same reason. Collate functions and
`Trainer`-style frameworks all carry a recursive `to_device`. What is stated here
about CUDA comes from the PyTorch 2.15 documentation (GPU operations are queued
asynchronously, and copying between the CPU and the GPU synchronizes); the CPU checks
were run, the GPU cases were not.
