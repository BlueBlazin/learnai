A LoRA adapter changes a linear layer by a low-rank update that is stored as two small
matrices. Implement the three operations a server needs: folding the adapter into the
weight, taking it out again, and running the layer with the adapter kept separate. The
formulas are one line each; the factor order and the scale are where implementations
go wrong.

## Specification

```python
def merge(w, a, b, alpha): ...  # -> merged weight, (out, in)
def unmerge(w_merged, a, b, alpha): ...  # -> original weight, (out, in)
def lora_forward(x, w, a, b, alpha): ...  # -> (..., out)
```

The base layer computes $y = x W^{\top}$ with a weight $W$ of shape `(out, in)`. A
LoRA adapter of rank $r$ consists of a **down-projection** $A$ of shape `(r, in)`, an
**up-projection** $B$ of shape `(out, r)`, and a number $\alpha$. It replaces the
weight by

$$
W' = W + \frac{\alpha}{r}\, B A .
$$

The rank $r$ is not an argument: it is the number of rows of `a` (equally, the number
of columns of `b`).

**`merge(w, a, b, alpha)`** returns $W' = W + \dfrac{\alpha}{r} B A$.

**`unmerge(w_merged, a, b, alpha)`** returns $W' - \dfrac{\alpha}{r} B A$, which
recovers the original weight when it is given the result of `merge` and the same
adapter.

**`lora_forward(x, w, a, b, alpha)`** returns the output of the adapted layer,
computed from the separate matrices:

$$
y = x W^{\top} + \frac{\alpha}{r}\, \big(x A^{\top}\big) B^{\top}.
$$

This equals $x W'^{\top}$, the output of the merged layer.

- `w`, `a`, `b`: float32 matrices with the shapes above, $1 \le r$. `b` may be all
  zeros, which is how adapters are initialized.
- `alpha`: a Python float. It may be zero or negative. It is not necessarily a
  multiple of $r$.
- `x`: a float32 tensor of shape `(..., in)` with one to three dimensions. The last
  axis is the input axis; all other axes are batch axes.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w`, `w_merged` | `(out, in)` | float32 |
| `a` | `(r, in)` | float32 |
| `b` | `(out, r)` | float32 |
| `x` | `(..., in)`, one to three dimensions | float32 |
| result of `merge`, `unmerge` | `(out, in)` | float32 |
| result of `lora_forward` | `(..., out)` | float32 |

## Example

```python
w = [[1.0, 0.0,  2.0],
     [0.0, 1.0, -1.0]]      # out = 2, in = 3
a = [[1.0, 0.0, 1.0],
     [0.0, 2.0, 0.0]]       # r = 2
b = [[1.0,  0.0],
     [2.0, -1.0]]
alpha = 1.0

merge(w, a, b, alpha)                       # [[1.5, 0.0, 2.5], [1.0, 0.0, 0.0]]
unmerge(merge(w, a, b, alpha), a, b, alpha) # [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]
lora_forward([1.0, 2.0, 3.0], w, a, b, alpha)  # [9.0, 1.0]
```

The scale is $\alpha / r = 1 / 2$. The update is

$$
B A = \begin{bmatrix} 1 & 0 \\ 2 & -1 \end{bmatrix}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 2 & 0 \end{bmatrix}
= \begin{bmatrix} 1 & 0 & 1 \\ 2 & -2 & 2 \end{bmatrix},
\qquad
W' = W + \tfrac12 B A = \begin{bmatrix} 1.5 & 0 & 2.5 \\ 1 & 0 & 0 \end{bmatrix}.
$$

For $x = [1, 2, 3]$ the base path gives $x W^{\top} = [7, -1]$. The adapter path
projects down to $r = 2$ numbers, $x A^{\top} = [4, 4]$, and back up,
$[4, 4]\, B^{\top} = [4, 4]$; scaled by $\tfrac12$ it adds $[2, 2]$. The result
$[9, 1]$ is the same as $x W'^{\top} = [1.5 + 7.5,\ 1]$.

## Constraints

- $1 \le \text{out}, \text{in} \le 64$ and $1 \le r \le 16$; $r$ may exceed neither,
  either, or both of `out` and `in`. `x` has at most 64 rows in total.
- Entries are of order 1 and $|\alpha| \le 64$.
- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-4`. `unmerge(merge(w, ...))` must reproduce `w` to the
  same tolerance: adding and subtracting the update in float32 is not exact, and that
  is expected.
- No Python loops are needed.

## Why this matters

A fine-tuned LoRA adapter is a few megabytes next to a base model of many gigabytes,
and a server has to decide what to do with it. Merged, the adapter costs nothing at
inference: the layer is an ordinary matmul with a different weight, which is how
single-adapter deployments and exported checkpoints work. Unmerged, the base weights
stay shared, so one copy of the model can serve many adapters in the same batch, at
the price of two small extra matmuls per layer. Switching adapters means unmerging one
and merging another, and that only works if all three operations agree on the same
$\frac{\alpha}{r} B A$. An adapter merged with the factors swapped or with the scale
off by $r$ does not crash. It produces a model whose fine-tuned behavior is too weak,
too strong, or scrambled, and the base model's fluency hides it.
