## The idea

A linear layer in `(out, in)` layout computes $y = x\,W^\top$, so output feature $y_i$ is the dot product of $x$ with row $i$ of $W$. Because rows correspond to output features, two projections of the same input ($W_0$ with $n_0$ rows and $W_1$ with $n_1$ rows) can be combined into one matrix with $n_0 + n_1$ rows:

$$
x \begin{bmatrix} W_0 \\ W_1 \end{bmatrix}^{\!\top} = \big[\, x\,W_0^\top \;\big|\; x\,W_1^\top \,\big].
$$

Fusing simply stacks rows. It does not reduce the number of multiplications, but one larger matrix product uses hardware more efficiently than several smaller ones, and one tensor is simpler to store.

Therefore, `fuse` concatenates weights along axis 0, and `split_fused` slices axis 0 using the cumulative sums of `parts`. Cutting the output of a fused projection along its last axis is equivalent to cutting the weight along axis 0. `swiglu_fused` uses this identity to run one matrix multiply and then split the resulting features into gate and up halves.

The tensor itself does not encode how many rows belong to each part or which part comes first. Those conventions come from the model code. Granite's `input_linear` places gate before up, while a fused `qkv_proj` places query, key, and value in order with widths determined by the head counts.

## Reference solution (PyTorch)

```python
def split_fused(w, parts):
    return list(torch.split(w, parts, dim=0))


def fuse(ws):
    return torch.cat(ws, dim=0)


def swiglu_fused(x, w_in, w_down):
    gate, up = (x @ w_in.T).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w_down.T
```

`torch.split` interprets a list argument as section sizes. In JAX and MLX, `split` interprets a list argument as cut positions instead, so those implementations slice with a running offset:

```python
out, start = [], 0
for size in parts:
    out.append(w[start : start + size])
    start += size
```

## Common mistakes

- **Cutting the input axis.** Calling `split(w, parts, dim=1)` cuts columns (input features) instead of rows. On a rectangular weight this raises a size error, but on a square weight, such as a fused matrix whose total output width equals the model width, it runs silently and returns pieces with transposed dimensions, like `(8, 4)` instead of `(4, 8)`. Fusing along axis 1 fails in the same way.
- **Confusing sizes with offsets.** `torch.tensor_split`, `np.split`, `jnp.split`, and `mx.split` interpret a list as split indices rather than sizes. Passing `[8, 4, 4]` cuts at positions 8, 4, and 4, yielding four pieces (two empty or mis-sliced) instead of three.
- **Splitting into equal chunks.** Calling `chunk(w, 3)` works for a QKV weight only when queries, keys, and values have equal width. Under grouped-query attention, the query projection is wider than the key and value projections.
- **Swapping gate and up.** Because $\operatorname{silu}(a) \cdot b$ is asymmetric, treating the first half as `up` changes the function. Where the gate is large and positive, $\operatorname{silu}(a)\,b \approx a\,b$, so the two readings nearly agree (and in the worked example, one output component even matches). Testing with negative gate values exposes the bug clearly.
- **Activating before splitting.** Computing `silu(x @ w_in.T)` and then splitting applies SiLU to the `up` half as well, which is wrong.

## Complexity

Let $n = \sum_i n_i$ be the total output width. `fuse` and `split_fused` copy or view $O(n \cdot d_{\text{in}})$ elements. `swiglu_fused` performs the same $O(h\,d)$ arithmetic per input vector as the unfused layer, replacing two matrix multiplies with one of twice the height for a constant-factor speedup.

## In the wild

Granite 4.0's MLP in Hugging Face Transformers (`GraniteMoeHybridMLP`) follows this exact pattern. Its `input_linear` outputs twice the hidden width, splits the result with `chunk(2, dim=-1)`, applies the activation to chunk 0, and multiplies by chunk 1. Phi-3 (`modeling_phi3.py`) uses both forms of fusion. Its `gate_up_proj` splits with `gate, up_states = up_states.chunk(2, dim=-1)`, and its `qkv_proj` slices first at `num_attention_heads * head_dim` and then by `num_key_value_heads * head_dim` to handle unequal grouped-query sizes. The `real-model-block` problem in this track runs the fused Granite layer against a real checkpoint. One common exception is GPT-2's fused `c_attn`, which is stored in `(in, out)` layout (transposed relative to the convention here) so its parts are blocks of columns rather than rows.
