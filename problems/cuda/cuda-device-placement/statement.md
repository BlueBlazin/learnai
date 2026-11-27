Implement a recursive helper `to_device` that moves a nested batch of tensors onto a target device, and a forward pass `run` that classifies padded token sequences using mean pooling without ever leaving that device. On a CPU, device-placement bugs go unnoticed because every tensor is already on the CPU. On a GPU, the same mistakes either raise a device-mismatch error or trigger slow, silent round trips through host memory.

## Specification

```python
def to_device(batch, device): ...
def run(weights, inputs, device): ...
```

### `to_device(batch, device)`

Recursively move every tensor inside a nested container to `device`.

- `batch` is a `torch.Tensor`, or a `dict`, `list`, or `tuple` nested to any depth. Its leaves may be tensors or other Python objects such as strings, numbers, or `None`.
- Return a new structure with the same container types. A `dict` produces a `dict` with the same keys in the same order, a `list` produces a `list`, and a `tuple` produces a `tuple`.
- Replace every tensor leaf with `tensor.to(device)`, and keep every non-tensor leaf as the exact same object.
- Do not mutate `batch`. Return new containers so the caller's original structure and tensors remain untouched.

### `run(weights, inputs, device)`

Compute classification logits for a padded batch of token sequences by mean-pooling each sequence's non-padding embeddings.

- `weights` is a `dict` of floating-point tensors already on `device`, all sharing the same dtype (`float32` or `float16`). It contains `"tok_emb"` (token embedding table), `"pos_emb"` (position embedding table), and `"w_out"` (output projection matrix).
- `inputs` is a nested `dict` whose tensors may start on any device (for example, on the CPU from a data loader):

  ```python
  {"tokens": ..., "lengths": ..., "head": {"bias": ...}}
  ```

  Here `tokens` holds a batch of $B$ token sequences padded to length $T$. For each row $b$, `lengths[b]` is the number of real tokens ($1 \le \texttt{lengths}[b] \le T$). Positions $t \ge \texttt{lengths}[b]$ are padding and must not affect the result.
- Return the classification logits on `device` with the same dtype as `weights`.

For each sequence $b$ and position $t$, add the token and position embeddings, average only over the $n_b = \texttt{lengths}[b]$ real positions, and apply the linear head. With $E_{\text{tok}} = \texttt{weights["tok\_emb"]}$, $E_{\text{pos}} = \texttt{weights["pos\_emb"]}$, $W = \texttt{weights["w\_out"]}$, and $\text{bias} = \texttt{inputs["head"]["bias"]}$:

$$
h_{b,t} = E_{\text{tok}}[\texttt{tokens}_{b,t}] + E_{\text{pos}}[t],
\qquad
\bar h_b = \frac{1}{n_b} \sum_{t < n_b} h_{b,t},
\qquad
\text{logits}_b = W\, \bar h_b + \text{bias}.
$$

### Placement rules

The judge tracks where your code allocates tensors and whether it reads device values back to the host.

1. **Allocate on `device`.** Every tensor created inside `run` must be created on `device`. Calling a factory function such as `torch.arange`, `torch.zeros`, `torch.ones`, `torch.full`, or `torch.tensor` without `device=` allocates on the CPU and fails the test.
2. **No host reads.** Do not read tensor values on the host during the forward pass. Avoid `.cpu()`, `.numpy()`, `.item()`, `.tolist()`, `int(x)`, `float(x)`, `bool(x)`, printing tensors, using a tensor as a slice bound or `range` argument, and boolean-mask indexing (`h[mask]`). Each of these forces the CPU to synchronize with the device.
3. **Preserve dtype.** Do not introduce hard-coded float dtypes into the computation. When `weights` is `float16`, the returned logits must be `float16`.

## Shapes

Here $B$ is the batch size, $T$ is the padded sequence length, $V$ is the vocabulary size, $P$ is the maximum position length ($P \ge T$), $D$ is the embedding dimension, and $C$ is the number of classes.

| Name | Shape | Dtype |
|---|---|---|
| `weights["tok_emb"]` | `(V, D)` | `float32` or `float16` |
| `weights["pos_emb"]` | `(P, D)` | same as `tok_emb` |
| `weights["w_out"]` | `(C, D)` | same as `tok_emb` |
| `inputs["tokens"]` | `(B, T)` | `int64` |
| `inputs["lengths"]` | `(B,)` | `int64` |
| `inputs["head"]["bias"]` | `(C,)` | same as `tok_emb` |
| `run` output | `(B, C)` | same as `tok_emb` |

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

- Row 0 has three real tokens. The combined embeddings are $[1, 0] + [0, 0]$, $[0, 1] + [0.5, 0.5]$, and $[1, 1] + [1, 1]$, with mean $[3.5, 3.5] / 3 = [1.1667, 1.1667]$. Adding the bias after projecting with $W$ gives $[1.1667 + 0.1,\; 1.1667 + 0.2,\; 0 + 0.3]$.
- Row 1 has one real token, with combined embedding $[1, 1] + [0, 0]$. The two padding positions are excluded, so the mean is $[1, 1]$ and the logits are $[1.1, 1.2, 0.3]$. Averaging over all three positions would incorrectly yield a mean of $[0.8333, 0.8333]$.

Calling `to_device` on `inputs` returns a new outer dict containing a new `"head"` dict and three tensors on `device`.

## Constraints

- $1 \le B \le 8$, $1 \le T \le 20$, and $P \ge T$. All token IDs lie in $[0, V)$.
- `float32` logits are compared against a `float64` reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. `float16` logits use a relative tolerance of `1e-2` and an absolute tolerance of `3e-2`.
- The placement rules are strict. Even one tensor created without `device=` or one host read fails the test, regardless of whether the numerical logits match.
- **CPU vs. GPU tests.** Tests without a hardware mark run on the CPU. They intercept factory calls and host reads made by your code, and one test runs `run` on `torch.device("meta")`. Like CUDA tensors, meta tensors hold shape and dtype without data, reject elementwise operations with CPU tensors, and cannot be read on the host. Two additional tests run only on an NVIDIA GPU and repeat these checks with CUDA weights and a CPU batch. The GPU tests were written from the PyTorch 2.15 documentation and could not be executed when this problem was authored.

## Why this matters

In production serving, tokenizers and data loaders produce nested dicts of CPU tensors while model weights stay on the GPU. Any helper tensor created inside the forward pass—such as position IDs, causal or padding masks, or token buffers—comes from a factory call. Omitting `device=` allocates it on the CPU.

At best, that triggers an immediate device-mismatch error. At worst, it fails silently. PyTorch allows indexing a CUDA tensor with a CPU index tensor, and working around errors with `.cpu()` forces host-device copies on every request while stalling the CPU until the GPU queue drains. Hard-coded dtypes cause similar silent regressions. Calling `.float()` on a mask promotes a `float16` forward pass to `float32` for every downstream layer, doubling activation memory.
