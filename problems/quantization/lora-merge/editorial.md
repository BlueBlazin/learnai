## The idea

LoRA (Hu et al., 2021) freezes a pretrained weight matrix $W$ and learns a weight update of rank at most $r$, factored as the product of two thin matrices:

$$
W' = W + \frac{\alpha}{r}\, B A, \qquad A \in \mathbb{R}^{r \times \text{in}},
\quad B \in \mathbb{R}^{\text{out} \times r}.
$$

Matrix $A$ projects the input down to $r$ dimensions, and matrix $B$ projects back up to the output dimension. When $r$ is 8 or 16 and layer widths are in the thousands, $A$ and $B$ hold well under 1 percent of the parameters of $W$.

Because the adapted layer remains linear, the merged and unmerged forward passes are algebraically identical:

$$
x W'^{\top} = x \Big(W + \frac{\alpha}{r} B A\Big)^{\top}
= x W^{\top} + \frac{\alpha}{r}\, x A^{\top} B^{\top}.
$$

The left side is the merged layer, requiring only one matrix multiplication. The right side is the unmerged layer, combining the base matrix multiplication with a low-rank detour through $r$ dimensions. Transposing a matrix product reverses its factor order, so the forward pass multiplies by $A^{\top}$ first and $B^{\top}$ second.

## Reference solution (PyTorch)

```python
def merge(w, a, b, alpha):
    return w + (alpha / a.shape[0]) * (b @ a)


def unmerge(w_merged, a, b, alpha):
    return w_merged - (alpha / a.shape[0]) * (b @ a)


def lora_forward(x, w, a, b, alpha):
    return x @ w.T + (alpha / a.shape[0]) * ((x @ a.T) @ b.T)
```

The JAX and MLX versions are identical apart from type hints.

**Why $\alpha / r$.** Scaling by $\alpha / r$ keeps the update magnitude roughly constant when changing the rank $r$, avoiding learning rate retuning. Because the scale is fixed during training, inference must apply the exact same value, which is why adapter checkpoints store both `r` and `lora_alpha`.

**Why `(x @ a.T) @ b.T`.** Matrix multiplication is associative, so `x @ (a.T @ b.T)` produces the same numbers. However, computing `a.T @ b.T` materializes a full `(in, out)` matrix on every forward call, defeating the purpose of keeping the adapter separate. Grouping as `(x @ a.T) @ b.T` reduces the adapter path to $O(r \cdot (\text{in} + \text{out}))$ work per input row.

**Why `unmerge` is not exact.** In float32, `(w + d) - d` matches `w` only up to rounding error because `w + d` rounds to the precision of the larger operand. A single merge-unmerge round trip loses roughly one part in $10^7$, which is harmless. Merging and unmerging thousands of times on the same weight tensor accumulates drift, so long-running servers either keep a pristine copy of the base weights or serve adapters unmerged.

## Common mistakes

- **Computing `a @ b` instead of `b @ a`.** The alphabetical order is tempting, and on a square layer the transposed product `a.T @ b.T` even matches the shape of `w` while producing the transposed update. Checking shapes—`(out, r) @ (r, in)`—always gives `b @ a`.
- **Scaling by `alpha` alone.** Omitting the division by `r` makes the update $r$ times too large. Under the common setting $\alpha = 2r$, the intended scale is 2, whereas scaling by `alpha` alone applies 32 at rank 16.
- **Applying the scale twice.** In a two-step forward pass, it is easy to scale the output of `a` and then scale the output of `b` again. This bug passes tests whenever $\alpha = r$ (a common default) because the scale is 1.
- **Using the wrong sign in `unmerge`.** Adding the update instead of subtracting doubles the adapter contribution. Forward-only tests miss this bug, while round-trip tests catch it immediately.
- **Reading the rank from the wrong axis.** `b.shape[0]` is `out` and `a.shape[1]` is `in`. The rank $r$ is the shared inner dimension `a.shape[0]` (or `b.shape[1]`).

## Complexity

`merge` and `unmerge` each perform one $(\text{out} \times r) \cdot (r \times \text{in})$ matrix multiplication, costing $O(\text{out} \cdot \text{in} \cdot r)$ once per adapter switch. `lora_forward` costs the base matrix multiplication plus $O(r \cdot (\text{in} + \text{out}))$ per input row, whereas the merged layer costs only the base matrix multiplication.

## In the wild

**Initialization.** LoRA initializes $A$ randomly and $B$ to all zeros so that the initial update is zero and the adapted model starts identical to the base model. That is why `b` may be all zeros and `merge` must return `w` unchanged in that case.

**PEFT.** Hugging Face PEFT stores `lora_A` and `lora_B` with these exact shapes as bias-free linear layer weights and computes `scaling = lora_alpha / r`. Its `merge` and `unmerge` methods add and subtract `scaling * (B @ A)`. With `use_rslora=True` (rank-stabilized LoRA), the scale changes to $\alpha / \sqrt{r}$, another reason the scaling convention must be stored alongside the adapter.

**Serving.** Merging is the right choice for single-adapter deployments because it adds zero inference overhead and allows the merged weight to be quantized normally. Multi-tenant servers such as S-LoRA and vLLM keep adapters unmerged so one copy of the base weights can batch requests for different adapters, running each request's low-rank path separately. Quantized base models add one more constraint: a full-precision adapter must either be merged before 4-bit quantization or kept as a separate floating-point path alongside the quantized base weight, as in QLoRA.
