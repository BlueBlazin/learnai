## The idea

A linear layer in `(out, in)` layout computes $y = x\,W^\top$: $y_i$ is the dot
product of $x$ with row $i$ of $W$. Rows are output features. Two projections of the
same input, $W_0$ with $n_0$ rows and $W_1$ with $n_1$ rows, can therefore be
evaluated by one matrix with $n_0 + n_1$ rows,

$$
x \begin{bmatrix} W_0 \\ W_1 \end{bmatrix}^{\!\top} = \big[\, x\,W_0^\top \;\big|\; x\,W_1^\top \,\big],
$$

and the outputs come out side by side. That is all fusing is: stacking rows. It
saves nothing in arithmetic, but one large matrix product is faster than several
small ones, and one tensor is simpler to store, which is why some code bases train
and ship the fused form.

So `fuse` concatenates along axis 0, `split_fused` cuts axis 0 at the running sum of
`parts`, and cutting the *output* of the fused projection along its last axis is
equivalent to cutting the weight along its first. The fused SwiGLU uses the second
form: one product, then split the features.

What the tensor does not tell you is the convention: how many rows belong to each
part, and in which order the parts are stacked. Both come from the model's code.
For Granite's `input_linear` the order is gate, then up; for a fused `qkv_proj` it is
query, key, value, with the sizes following from the head counts.

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

`torch.split` with a list takes sizes. In JAX and MLX, `split` with a list takes cut
positions, so those references slice with a running offset instead:

```python
out, start = [], 0
for size in parts:
    out.append(w[start : start + size])
    start += size
```

## Common mistakes

- **Cutting the input axis.** `split(w, parts, dim=1)` cuts columns, which are input
  features. For a rectangular weight it fails on the sizes; for a square one, such as
  a fused weight whose total output width equals the model width, it runs and
  returns pieces with the two dimensions exchanged, `(8, 4)` where `(4, 8)` was
  meant. Fusing along axis 1 has the same symptom.
- **Sizes versus offsets.** `tensor_split`, `np.split`, `jnp.split`, and `mx.split`
  read a list as the positions of the cuts. `[8, 4, 4]` then means "cut at 8, at 4,
  and at 4": four pieces, two of them empty or overlapping, instead of three.
- **Equal chunks.** `chunk(w, 3)` is right for a QKV weight only when queries, keys,
  and values have the same width. With grouped-query attention the query part is
  wider than the other two.
- **Gate and up exchanged.** $\operatorname{silu}(a) \cdot b$ is not symmetric, so
  taking the first half as up changes the result, but only through the activation:
  where the gate is large and positive, $\operatorname{silu}(a)\,b \approx a\,b$ and
  the two readings nearly agree. In the worked example one of the two outputs is even
  identical. A test with a negative gate branch separates them clearly.
- **Activating before splitting.** `silu(x @ w_in.T)` followed by a split applies the
  activation to the up half as well. It is one line shorter and wrong.

## Complexity

`fuse` and `split_fused` copy or view $O(n \cdot d_{in})$ numbers. `swiglu_fused`
costs the same $O(h\,d)$ per input vector as the unfused layer; the fused form
replaces two matrix products with one of twice the height, which is a constant-factor
gain from better batching, not a change in the amount of arithmetic.

## In the wild

Granite 4.0's MLP in Hugging Face Transformers (`GraniteMoeHybridMLP`) is this layer:
`input_linear` produces twice the hidden width, the result is cut with
`chunk(2, dim=-1)`, and the activation is applied to chunk 0, which then multiplies
chunk 1. Phi-3 (`modeling_phi3.py`) does both kinds of fusing: `gate_up_proj` with
`gate, up_states = up_states.chunk(2, dim=-1)`, and `qkv_proj`, whose output is sliced
at `num_attention_heads * head_dim` and then at a further `num_key_value_heads *
head_dim`, so its three parts are unequal under grouped-query attention.
`real-model-block` in this track runs the fused Granite layer against the real
checkpoint. One layout this problem leaves out: GPT-2's fused `c_attn` is stored
`(in, out)`, transposed relative to everything here, so there the parts are blocks of
columns.
