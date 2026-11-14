## The idea

LoRA (Hu et al., 2021) freezes a pretrained weight $W$ and learns a change to it that
is constrained to have rank at most $r$, by writing the change as a product of two
thin matrices:

$$
W' = W + \frac{\alpha}{r}\, B A, \qquad A \in \mathbb{R}^{r \times \text{in}},
\quad B \in \mathbb{R}^{\text{out} \times r}.
$$

$A$ projects the input down to $r$ numbers, and $B$ projects those back up to the
output. With $r$ of 8 or 16 and layer widths in the thousands, the two factors hold
well under one percent of the parameters of $W$.

Because the adapted layer is still linear, there are two equivalent ways to run it,
and the whole problem is that equivalence:

$$
x W'^{\top} = x \Big(W + \frac{\alpha}{r} B A\Big)^{\top}
= x W^{\top} + \frac{\alpha}{r}\, x A^{\top} B^{\top}.
$$

The left side is the merged layer: one matmul, no trace of the adapter. The right side
is the unmerged layer: the base matmul plus a detour through $r$ dimensions. The
transpose of a product reverses the factors, which is why the forward pass applies
$A^{\top}$ first and $B^{\top}$ second, the order in which the data flows.

## Reference solution (PyTorch)

```python
def merge(w, a, b, alpha):
    return w + (alpha / a.shape[0]) * (b @ a)


def unmerge(w_merged, a, b, alpha):
    return w_merged - (alpha / a.shape[0]) * (b @ a)


def lora_forward(x, w, a, b, alpha):
    return x @ w.T + (alpha / a.shape[0]) * ((x @ a.T) @ b.T)
```

The JAX and MLX versions are identical apart from the type hints.

**Why $\alpha / r$.** The paper scales the update by $\alpha / r$ so that changing the
rank does not require retuning the learning rate: $\alpha$ is set once, and the
division compensates for the update growing with the number of rank-one terms. The
scale belongs to the adapter. It is fixed when training starts, and inference has to
use the same value, which is why adapter files store both `r` and `lora_alpha`.

**Why `(x @ a.T) @ b.T`.** Matrix multiplication is associative, so
`x @ (a.T @ b.T)` gives the same numbers. It also builds an `(in, out)` matrix on
every call, which is the full-size update the adapter exists to avoid. Grouped the
other way, the adapter path costs $O(r \cdot (\text{in} + \text{out}))$ per input row.

**Why `unmerge` is not exact.** In float32, `(w + d) - d` is `w` only up to rounding:
the sum is rounded to the precision of the larger of the two operands. One round trip
loses about one part in $10^7$, which is harmless. A server that merges and unmerges
adapters thousands of times on the same weight tensor accumulates that drift, so
long-running servers keep a pristine copy of the base weights or never merge at all.

## Common mistakes

- **`a @ b` for `b @ a`.** The names suggest alphabetical order, and for a square
  layer with the factors transposed (`a.T @ b.T`) the shapes even work out: the result
  is the transpose of the update. The shape rule settles it: `(out, r) @ (r, in)`.
- **Scaling by `alpha` alone.** Dropping the division makes the update $r$ times too
  strong. With the common setting $\alpha = 2r$ the correct scale is 2, and the
  mistaken one is 32 at rank 16.
- **Applying the scale twice.** In a forward pass written in two steps, it is easy to
  scale the output of `a` and then scale the output of `b` again. The result is right
  exactly when $\alpha = r$, a common default, which is how this bug survives testing.
- **`unmerge` with the wrong sign.** Adding again leaves a weight with twice the
  update. Round-trip tests catch it; forward-only tests never call `unmerge`.
- **Reading the rank from the wrong axis.** `b.shape[0]` is `out`, and `a.shape[1]` is
  `in`. The rank is the shared inner dimension, `a.shape[0] == b.shape[1]`.

## Complexity

`merge` and `unmerge` cost one $(\text{out} \times r) \cdot (r \times \text{in})$
product, $O(\text{out} \cdot \text{in} \cdot r)$, once per adapter switch.
`lora_forward` costs the base matmul plus $O(r \cdot (\text{in} + \text{out}))$ per
input row; the merged layer costs the base matmul only.

## In the wild

**Initialization.** The paper initializes $A$ randomly and $B$ to zero, so the update
starts at zero and the adapted model starts out identical to the base model. That is
why `b` may be all zeros here, and `merge` must then return `w` unchanged.

**PEFT.** Hugging Face PEFT stores `lora_A` and `lora_B` with exactly these shapes
(as the weights of two bias-free linear layers) and a `scaling` of `lora_alpha / r`.
Its `merge` and `unmerge` add and subtract `scaling * (B @ A)`. With
`use_rslora=True` the scale is $\alpha / \sqrt{r}$ (rank-stabilized LoRA), a different
convention for the same slot, and one more reason the scale has to travel with the
adapter.

**Serving.** Merging is the right choice for one adapter per deployment: zero
overhead, and the merged weight can be quantized like any other. Multi-tenant servers
keep adapters unmerged and batch requests for different adapters over one copy of the
base weights, applying each request's low-rank path separately (S-LoRA; the LoRA
support in vLLM). The interaction with quantization is the catch: an adapter trained
against full-precision weights and merged into a 4-bit model has to be merged before
quantizing, or kept as a separate float path next to the quantized base, as QLoRA
does during training.
