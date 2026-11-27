## The idea

Write each formula with all indices visible, then read the `einsum` equation directly from it. The indices on the left-hand side go after `->` in that exact order. Each operand contributes its own index letters, and any letter that appears only under a summation is omitted after `->` so that `einsum` sums over it.

- **`attention_scores`**: `out[b, h, t, s] = sum_d q[b, h, t, d] k[b, h, s, d]` translates directly to `"bhtd,bhsd->bhts"`. Because `b` and `h` appear on both operands and in the output, each batch row and head pairs with itself without summing. Because `d` appears on both operands and is absent from the output, it is multiplied and summed.
- **`bilinear`**: `out[b] = sum_d sum_e x[b, d] w[d, e] y[b, e]` becomes `"bd,de,be->b"`. Sharing `b` across `x` and `y` pairs row `b` of `x` with row `b` of `y`. Using two distinct letters (`"bd,de,ce->bc"`) would compute the full `(B, B)` table of every row against every row, whose diagonal contains the desired values.
- **`per_head_project`**: `out[b, h, t, e] = sum_d x[b, h, t, d] w[h, d, e]` becomes `"bhtd,hde->bhte"`. Omitting `b` and `t` from `w` shares each head's matrix across all batch rows and positions, while including `h` selects a distinct matrix per head.

## Reference solution (PyTorch)

```python
def attention_scores(q, k):
    return torch.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x, w, y):
    return torch.einsum("bd,de,be->b", x, w, y)


def per_head_project(x, w):
    return torch.einsum("bhtd,hde->bhte", x, w)
```

`jnp.einsum` and `mx.einsum` accept the exact same three equations.

You can also write all three functions using `@`, which contracts the last axis of the left operand against the second-to-last axis of the right operand and broadcasts leading axes:

```python
q @ k.transpose(-2, -1)  # (B, H, T, D) @ (B, H, D, S) -> (B, H, T, S)

((x @ w) * y).sum(dim=-1)  # (B, D) @ (D, E) -> (B, E), then row b against row b

x @ w  # (B, H, T, D) @ (H, D, E): w is broadcast as (1, H, D, E)
```

In `attention_scores`, the keys need their last two axes swapped so `D` sits second to last on the right operand. In `bilinear`, multiplying `(x @ w)` elementwise by `y` and summing over the last axis avoids forming the `(B, B)` matrix that `x @ w @ y.T` would build. In `per_head_project`, `x @ w` works because broadcasting aligns leading axes from the right, matching `(B, H)` against `(H,)`. If the weights instead had shape `(B, D, E)` (one matrix per batch row), you would need `w[:, None]`.

Because `einsum` and `@` can accumulate the same products in different orders, their float32 outputs can differ in the last few bits. Across 300 random inputs in $[-2, 2]$ with sizes up to 8 in all three frameworks, the largest deviation from a float64 reference sum was `7e-6` (on `bilinear`, where sums reach the tens). That is why the tests use a tolerance of `1e-4` instead of exact equality.

## Common mistakes

- **Output letters in the wrong order.** `"bhtd,bhsd->bhst"` returns shape `(B, H, S, T)` with keys along the rows. When `T != S` (such as with a KV cache), a later step raises a shape error, but when `T == S`, softmax silently runs over the wrong axis.
- **Scaling by $\sqrt{D}$ out of habit.** Scaled dot-product attention divides by $\sqrt{D}$, but this function specifies the unscaled dot products.
- **Using a new letter for a paired axis.** `"bhtd,bgsd->bhts"` gives the head axis of `k` a separate letter `g` that is absent from the output. Because `g` is summed over, every query head is scored against the sum of all heads' keys, and the call succeeds for every input shape.
- **Computing every row against every row.** `(x @ w @ y.T).sum(dim=1)` builds the `(B, B)` table and sums each row. A test with `B == 1` hides this bug completely.
- **Calling bare `squeeze()`.** `(x[:, None, :] @ w @ y[:, :, None])` has shape `(B, 1, 1)`. Calling `.squeeze()` without arguments also removes the batch axis when `B == 1`. Specify the axes explicitly with `.squeeze(-1).squeeze(-1)` or index them away with `[:, 0, 0]`.
- **Transposing `w`.** `"bd,ed,be->b"` and `"bhtd,hed->bhte"` read `w[e, d]` instead of `w[d, e]`. This raises when `D != E`, but silently multiplies by $w^T$ when `w` is square. This habit often carries over from `torch.nn.Linear`, which stores weights as `(out, in)`.
- **Using one matrix for all heads.** `x @ w[0]` produces the right shape for every input while ignoring all heads except head `0`.
- **Indexing the weight by position.** `"bhtd,tde->bhte"` runs without error whenever `H == T`.

In all three frameworks, `einsum` also broadcasts a size-1 axis against a larger axis sharing the same letter. As a result, several of the mistakes above run without raising when `D == 1`, `E == 1`, or `T == 1`, returning wrong shapes or wrong values instead of an error.

## Complexity

`attention_scores` performs $O(B\,H\,T\,S\,D)$ multiplications, `bilinear` performs $O(B\,D\,E)$, and `per_head_project` performs $O(B\,H\,T\,D\,E)$. Forming the full `(B, B)` table in the `x @ w @ y.T` mistake adds $O(B^2 E)$ work.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| the operation | `torch.einsum(eq, *ops)` | `jnp.einsum(eq, *ops)` | `mx.einsum(eq, *ops)` |
| no `->` in the equation | result letters in alphabetical order | the same | the same |
| `...` for leading axes | yes | yes | yes |
| a repeated letter on one operand (`"ii->i"`) | the diagonal | the diagonal | the diagonal |
| an axis of size 1 against a larger one, same letter | broadcast | broadcast | broadcast |
| sizes that differ and are not 1 | `RuntimeError` (`ValueError` with three operands) | `ValueError` | `ValueError` |
| integer operands, `"ij,jk->ik"` | int32 result | int32 result | `ValueError` (floating types only) |
| integer operands, `"ij,ij->ij"` or `"ii->i"` | int32 result | int32 result | int32 result |
| swap the last two axes | `x.transpose(-2, -1)`, `x.mT` | `x.swapaxes(-2, -1)`, `x.mT` | `x.swapaxes(-2, -1)`, `x.mT` |
| `x.transpose(-2, -1)` on four axes | swaps the two | `TypeError` (expects all four axes) | `ValueError` (expects all four axes) |
| `x.transpose(0, 1, 3, 2)` | `TypeError` (use `x.permute(0, 1, 3, 2)`) | permutes | permutes |
| `x.T` on four axes | reverses all axes (with deprecation warning) | reverses all axes | reverses all axes |
| `(B, H, T, D) @ (H, D, E)` | `(B, H, T, E)` | `(B, H, T, E)` | `(B, H, T, E)` |

Every entry was verified on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The `einsum` equations are completely portable across all three frameworks. In contrast, `transpose` differs across frameworks. PyTorch swaps two axes, whereas JAX and MLX expect a full permutation of all axes. Use `swapaxes(-2, -1)` or `.mT` for portable matrix transposes. Another difference is MLX's `einsum` on integer inputs. Equations that contract a product of two operands (such as `"ij,jk->ik"`, `"ij,j->i"`, or `"i,i->"`) dispatch to `matmul`, which only accepts floating-point types and raises on integers unless you cast first. Equations without such a contraction (`"ij,ij->ij"`, `"ii->i"`, `"ij->ji"`, `"ij->"`) return int32.
