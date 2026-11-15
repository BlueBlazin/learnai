Some checkpoints store several linear projections of the same input as one matrix:
query, key, and value in a single `qkv_proj`, or the gate and up projections of SwiGLU
in a single `input_linear`. Implement the conversion in both directions, and the
SwiGLU layer that reads the fused form directly. Which axis is cut, and which block
comes first, decide whether the result is the same model or a different one of the
same shape.

## Specification

```python
def split_fused(w, parts): ...
def fuse(ws): ...
def swiglu_fused(x, w_in, w_down): ...
```

Weights are in `(out, in)` layout, the layout of `nn.Linear.weight` and of
checkpoints: a projection with weight $W$ of shape `(n, d_in)` maps $x$ to
$x\,W^\top$, and row $i$ of $W$ produces output feature $i$.

A **fused** weight computes $k$ projections of the same input at once. Its output
features are the outputs of projection 0, then those of projection 1, and so on:

$$
x\,W_{\text{fused}}^\top = \big[\; x\,W_0^\top \;\big|\; x\,W_1^\top \;\big|\; \cdots \;\big|\; x\,W_{k-1}^\top \;\big]
\qquad \text{(concatenated along the last axis).}
$$

**`fuse(ws)`** returns the fused weight $W_{\text{fused}}$ for the list of weights
`ws` $= [W_0, \dots, W_{k-1}]$, where $W_i$ has shape `(n_i, d_in)`. The result has
shape `(n_0 + ... + n_{k-1}, d_in)`. The list is not empty.

**`split_fused(w, parts)`** is the inverse. `parts` $= [n_0, \dots, n_{k-1}]$ lists
the output width of each projection, in order: these are **sizes**, not offsets, they
are positive, and they sum to the number of rows of `w`. The function returns a list
of $k$ tensors, the $i$-th of shape `(n_i, d_in)`, such that fusing them again gives
`w`. The sizes need not be equal.

**`swiglu_fused(x, w_in, w_down)`** is the SwiGLU layer of `swiglu-mlp` with its two
input projections fused, **gate first**: `w_in` has shape `(2h, d)` and equals
`fuse([w_gate, w_up])`.

$$
z = x\,W_{\text{in}}^\top, \qquad
\text{gate} = z_{\,0 \,..\, h-1}, \quad \text{up} = z_{\,h \,..\, 2h-1}, \qquad
y = \big(\operatorname{silu}(\text{gate}) \odot \text{up}\big)\, W_{\text{down}}^\top ,
$$

with $\operatorname{silu}(a) = a \cdot \sigma(a)$. Only the gate half goes through the
activation. There are no biases.

- `x`: float32, shape `(..., d)`, with one to three dimensions.
- `w_in`: float32, shape `(2h, d)`. `w_down`: float32, shape `(d_out, h)`.
- Returns float32, shape `(..., d_out)`.

You may use your framework's `silu` or `sigmoid`, and its split and concatenate
functions.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(n_0 + ... + n_{k-1}, d_in)` | float32 |
| `parts` | a list of $k$ positive ints | |
| result of `split_fused` | a list of $k$ tensors, `(n_i, d_in)` | float32 |
| `ws` | a list of $k$ tensors, `(n_i, d_in)` | float32 |
| result of `fuse` | `(n_0 + ... + n_{k-1}, d_in)` | float32 |
| `x` | `(..., d)`, one to three dimensions | float32 |
| `w_in` | `(2h, d)` | float32 |
| `w_down` | `(d_out, h)` | float32 |
| result of `swiglu_fused` | `(..., d_out)` | float32 |

## Example

```python
w = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  # shape (3, 2): three output features

split_fused(w, [1, 2])  # [[[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]]
fuse([[[1.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]])  # w
```

The first projection has one output feature (row 0), the second has two (rows 1 and
2). For $x = [1, 1]$: $x\,W^\top = [3, 7, 11]$, which is $[3]$ followed by $[7, 11]$.

The layer of the `swiglu-mlp` example, fused ($d = 2$, $h = 3$, $d_{out} = 2$):

```python
x = [1.0, 2.0]
w_in = [
    [1.0, 0.0],
    [0.0, -1.0],
    [1.0, 1.0],  # gate rows
    [1.0, 1.0],
    [2.0, 0.0],
    [0.0, 0.5],
]  # up rows; shape (6, 2)
w_down = [[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]  # shape (2, 3)

swiglu_fused(x, w_in, w_down)  # [5.050899, -3.334534]
```

$z = x\,W_{\text{in}}^\top = [1, -2, 3, 3, 2, 1]$, so the gate is $[1, -2, 3]$ and up
is $[3, 2, 1]$. Then $\operatorname{silu}(\text{gate}) \odot \text{up} =
[2.193176, -0.476812, 2.857723]$ and $y = [5.050899, -3.334534]$. Reading the halves
the other way round gives $[5.050899, -5.716364]$: the same shape, a first component
that happens to agree, and a different function.

## Constraints

- `split_fused` and `fuse` only move numbers and are compared exactly.
  `swiglu_fused` is compared with the reference, and with a three-matrix SwiGLU on
  the split weights, to a relative tolerance of `1e-4` and an absolute tolerance of
  `1e-5`.
- $1 \le k \le 8$ parts; $1 \le n_i, d_{in}, d, h, d_{out} \le 64$. A fused weight may
  be square.
- Pre-activations of the gate lie in $[-100, 100]$, and the result must be finite.
- No Python loops over rows or features are needed (a loop over the $k$ parts is
  fine).

## Why this matters

Fusing is a storage and speed choice that different code bases make differently for
the same architecture. Granite 4.0 350M ships the gate and up projections of every
MLP as one `shared_mlp.input_linear.weight` of shape `(4096, 1024)`; Phi-3 ships
`qkv_proj` and `gate_up_proj`; Llama-style checkpoints ship every projection
separately. An engine that supports one form has to convert the other when it loads
the file, and a quantization or LoRA tool that addresses `q_proj` by name has to
find it inside `qkv_proj`. Nothing in the tensor says where the cuts are or what
comes first. With grouped-query attention the three parts of `qkv_proj` do not even
have the same size, so cutting into equal thirds produces three matrices that load
and run. The two halves of a fused SwiGLU input have identical shapes, and swapping
them changes the output only through the activation, which is exactly the kind of
error that passes a shape check and a smoke test.
