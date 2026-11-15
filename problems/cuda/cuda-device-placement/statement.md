Move a nested batch to a device, then compute a small forward pass on it that needs
two tensors of its own (position ids and a padding mask) without ever leaving that
device. On a CPU nothing can go wrong, because everything is on the CPU; the same
code on a GPU either raises "Expected all tensors to be on the same device" or, worse,
quietly makes a round trip through host memory in the middle of the forward pass.

## Specification

```python
def to_device(batch, device): ...
def run(weights, inputs, device): ...
```

**`to_device(batch, device)`** moves every tensor of a nested structure:

- `batch`: a tensor, or a `dict`, `list`, or `tuple`, nested to any depth, whose
  leaves are tensors and other objects (strings, numbers, `None`).
- Returns a structure of the same form: a dict gives a dict with the same keys in the
  same order, a list a list, a tuple a tuple. Every tensor leaf is replaced by
  `tensor.to(device)`; every other leaf is passed through as the same object.
- `batch` is not modified: new containers are returned, and the caller's structure
  still holds its original tensors.

**`run(weights, inputs, device)`** classifies a padded batch of token sequences by
mean pooling:

- `weights`: a dict of float tensors that are **already on `device`**, all of one
  dtype (float32 or float16): `"tok_emb"` of shape `(V, D)`, `"pos_emb"` of shape
  `(P, D)`, `"w_out"` of shape `(C, D)`.
- `inputs`: a nested dict whose tensors may be on any device (in practice the CPU,
  fresh from the data loader):

  ```python
  {"tokens": ..., "lengths": ..., "head": {"bias": ...}}
  ```

  `tokens` holds `B` sequences padded to `T` positions; `lengths[b]` is the number
  of real tokens in row `b`, $1 \le \texttt{lengths}[b] \le T$, and positions
  $t \ge \texttt{lengths}[b]$ are padding whose token ids mean nothing. `bias` has
  the dtype of the weights.
- Returns the logits, shape `(B, C)`, in the dtype of the weights, on `device`.

With $E$ = `tok_emb`, $P$ = `pos_emb`, $W$ = `w_out`, and $n_b$ = `lengths[b]`:

$$
h_{b,t} = E[\texttt{tokens}_{b,t}] + P[t],
\qquad
\bar h_b = \frac{1}{n_b} \sum_{t < n_b} h_{b,t},
\qquad
\text{logits}_b = W\, \bar h_b + \text{bias}.
$$

Padding positions do not contribute to the mean.

**Placement rules.** The judge observes where your code puts tensors:

1. Every tensor `run` creates is created on `device`. A factory function
   (`torch.arange`, `torch.zeros`, `torch.ones`, `torch.full`, `torch.tensor`, ...)
   called without `device=` puts its result on the default device, the CPU; such a
   call fails the case.
2. Nothing is read on the host during the forward pass: no `.cpu()`, `.numpy()`,
   `.item()`, `.tolist()`, `int(x)`, `float(x)`, `bool(x)` on a tensor, no tensor
   used as a slice bound or in `range`, no boolean-mask indexing (`h[mask]`), no
   printing. Each of these makes the host wait for the device.
3. No float tensor of a hard-coded dtype enters the computation: with float16
   weights the result is float16.

## Shapes

| name | shape | dtype |
|---|---|---|
| `weights["tok_emb"]` | `(V, D)` | float32 or float16 |
| `weights["pos_emb"]` | `(P, D)`, $P \ge T$ | the same |
| `weights["w_out"]` | `(C, D)` | the same |
| `inputs["tokens"]` | `(B, T)` | int64 |
| `inputs["lengths"]` | `(B,)` | int64 |
| `inputs["head"]["bias"]` | `(C,)` | the dtype of the weights |
| result of `run` | `(B, C)` | the dtype of the weights |

## Example

```python
weights = {
    "tok_emb": [[0, 0], [1, 0], [0, 1], [1, 1]],  # V = 4, D = 2
    "pos_emb": [[0, 0], [0.5, 0.5], [1, 1]],  # P = 3
    "w_out": [[1, 0], [0, 1], [1, -1]],  # C = 3
}
inputs = {
    "tokens": [[1, 2, 3], [3, 0, 0]],
    "lengths": [3, 1],
    "head": {"bias": [0.1, 0.2, 0.3]},
}

run(weights, inputs, torch.device("cpu"))
# [[1.2667, 1.3667, 0.3],
#  [1.1,    1.2,    0.3]]
```

- Row 0 has three real tokens: $h = [1, 0] + [0, 0]$, $[0, 1] + [0.5, 0.5]$,
  $[1, 1] + [1, 1]$, whose mean is $[3.5, 3.5] / 3 = [1.1667, 1.1667]$. The logits
  are $[1.1667 + 0.1,\; 1.1667 + 0.2,\; 0 + 0.3]$.
- Row 1 has one real token: $h_0 = [1, 1] + [0, 0]$. The two padding positions are
  ignored, so the mean is $[1, 1]$ and the logits are $[1.1, 1.2, 0.3]$. Averaging
  over all three positions would give $[0.8333, 0.8333]$ instead.

`to_device` on the same `inputs` returns a new dict with a new inner dict, and three
tensors on the device.

## Constraints

- $1 \le B \le 8$, $1 \le T \le 20$, $P \ge T$; token ids are in $[0, V)$.
- float32 logits are compared with a float64 computation to a relative tolerance of
  `1e-4` and an absolute tolerance of `1e-5`; float16 logits to `1e-2` and `3e-2`.
- The placement rules are exact: one tensor created without `device=`, or one host
  read, fails the case even when the logits are right.
- **What runs where.** The cases without a hardware mark run on the CPU. They watch
  the factory functions and the host reads made by your file, and one of them runs
  `run` on `torch.device("meta")`, a device whose tensors have a shape and no data:
  like a CUDA tensor, a meta tensor refuses elementwise operations with a CPU tensor,
  and it cannot be read at all. Two cases run only on a machine with an NVIDIA GPU
  and repeat the checks with CUDA weights and a CPU batch. The GPU cases were written
  from the PyTorch 2.15 documentation and could not be run when the problem was
  authored.

## Why this matters

Batches reach a model server from the tokenizer and the data loader, on the CPU, as
nested dicts; weights live on the GPU. Everything the forward pass builds for itself
(position ids, causal and padding masks, a buffer for generated tokens) is created by
a factory call, and a factory call without `device=` creates on the CPU. The lucky
outcome is an exception at the first elementwise operation. The unlucky ones are
silent: indexing a CUDA tensor with a CPU index tensor works, and code that "fixes"
the exception with `.cpu()` now copies activations to host memory and back on every
request, with the host stalled until the GPU has drained its queue. The same holds
for dtypes: one `.float()` on a mask turns a half-precision forward pass into a
single-precision one from that point on, doubling the memory of every later
activation.
