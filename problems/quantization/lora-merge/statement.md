A LoRA adapter modifies a linear layer with a low-rank weight update stored as two small matrices. Implement the three operations needed to serve an adapted layer: merging the adapter into the base weight, unmerging it back out, and running the forward pass with the adapter kept separate.

## Specification

```python
def merge(w, a, b, alpha): ...  # -> merged weight, (out, in)
def unmerge(w_merged, a, b, alpha): ...  # -> original weight, (out, in)
def lora_forward(x, w, a, b, alpha): ...  # -> (..., out)
```

A base linear layer maps an input tensor $x$ to $y = x W^{\top}$ using a weight matrix $W$ of shape `(out, in)`. A LoRA adapter of rank $r$ consists of a **down-projection** matrix $A$ of shape `(r, in)`, an **up-projection** matrix $B$ of shape `(out, r)`, and a scalar $\alpha$. The adapter replaces the base weight with

$$
W' = W + \frac{\alpha}{r}\, B A .
$$

The rank $r$ is not passed as an argument. Read it from the number of rows of `a` (equivalently, the number of columns of `b`).

**`merge(w, a, b, alpha)`** computes the merged weight $W' = W + \dfrac{\alpha}{r} B A$.

**`unmerge(w_merged, a, b, alpha)`** computes $W' - \dfrac{\alpha}{r} B A$, recovering the original weight when given the output of `merge` and the same adapter.

**`lora_forward(x, w, a, b, alpha)`** computes the adapted layer's output without forming the full merged weight matrix:

$$
y = x W^{\top} + \frac{\alpha}{r}\, \big(x A^{\top}\big) B^{\top}.
$$

This produces the same output as $x W'^{\top}$.

- `a`, `b`: rank $r \ge 1$. Matrix `b` may be all zeros, which is how LoRA adapters are initialized.
- `alpha`: a Python float. It may be zero or negative, and it is not necessarily a multiple of $r$.
- `x`: the last axis is the input dimension `in`, and all leading axes are batch dimensions.

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

With $\alpha = 1.0$ and $r = 2$, the scale factor is $\alpha / r = 1 / 2$. Multiplying $B$ and $A$ gives the update:

$$
B A = \begin{bmatrix} 1 & 0 \\ 2 & -1 \end{bmatrix}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 2 & 0 \end{bmatrix}
= \begin{bmatrix} 1 & 0 & 1 \\ 2 & -2 & 2 \end{bmatrix},
\qquad
W' = W + \tfrac12 B A = \begin{bmatrix} 1.5 & 0 & 2.5 \\ 1 & 0 & 0 \end{bmatrix}.
$$

For $x = [1, 2, 3]$, the base path gives $x W^{\top} = [7, -1]$. The adapter path projects down to $r = 2$ values with $x A^{\top} = [4, 4]$, projects back up with $[4, 4]\, B^{\top} = [4, 4]$, and scales by $\tfrac12$ to add $[2, 2]$. The final output $[9, 1]$ matches $x W'^{\top} = [1.5 + 7.5,\ 1]$.

## Constraints

- $1 \le \text{out}, \text{in} \le 64$ and $1 \le r \le 16$. The rank $r$ may be smaller than, equal to, or larger than `out` or `in`. `x` has at most 64 rows in total.
- Tensor entries are of order 1 and $|\alpha| \le 64$.
- Results are compared with the reference to a relative tolerance of `1e-4` and an absolute tolerance of `1e-4`. `unmerge(merge(w, ...))` must reproduce `w` to the same tolerance, since adding and subtracting the update in float32 is not exact.
- No Python loops are needed.

## Why this matters

A fine-tuned LoRA adapter takes only a few megabytes compared to a multi-gigabyte base model, giving serving systems two deployment choices. Merging the adapter into $W$ eliminates all inference overhead because the layer reduces to a single standard matrix multiplication, which is ideal for single-adapter deployments and exported checkpoints. Keeping the adapter unmerged lets one shared copy of the base weights serve many adapters in the same batch at the cost of two small extra matrix multiplications per layer.

Switching adapters in place requires unmerging one adapter and merging the next, so all three operations must agree on the exact same $\frac{\alpha}{r} B A$ update. Swapping the factor order or missing the division by $r$ does not raise an error. Instead, it silently produces a model whose adapted behavior is too weak, too strong, or scrambled while the base model's fluency masks the bug.
