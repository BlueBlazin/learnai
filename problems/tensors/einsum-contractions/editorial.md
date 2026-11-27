## The idea

Write the formula with every index visible, then read the `einsum` equation off it.
The letters on the left-hand side of the formula go after `->`, in that order. Each
operand contributes its own letters. Whatever appears only under the sum is left out
of the result, and that is the entire instruction to sum over it.

**`attention_scores`**: `out[b, h, t, s] = sum_d q[b, h, t, d] k[b, h, s, d]` is
`"bhtd,bhsd->bhts"`. `b` and `h` are on both operands and in the result: batch row
with batch row, head with head, nothing summed. `d` is on both operands and absent
from the result: multiplied and summed.

**`bilinear`**: `out[b] = sum_d sum_e x[b, d] w[d, e] y[b, e]` is `"bd,de,be->b"`.
The `b` shared by `x` and `y` is what keeps row `b` of one with row `b` of the other.
With two different letters, `"bd,de,ce->bc"`, the same code returns the `(B, B)`
table of every row against every row, and the wanted values are its diagonal.

**`per_head_project`**: `out[b, h, t, e] = sum_d x[b, h, t, d] w[h, d, e]` is
`"bhtd,hde->bhte"`. `w` has no `b` and no `t`, so the same matrix is used at every
batch row and position, and it has `h`, so each head has its own.

## Reference solution (PyTorch)

```python
def attention_scores(q, k):
    return torch.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x, w, y):
    return torch.einsum("bd,de,be->b", x, w, y)


def per_head_project(x, w):
    return torch.einsum("bhtd,hde->bhte", x, w)
```

`jnp.einsum` and `mx.einsum` take the same three equations.

The same functions with `@`, which sums the last axis of its left operand against the
second-to-last axis of its right operand and broadcasts the axes before them:

```python
q @ k.transpose(-2, -1)  # (B, H, T, D) @ (B, H, D, S) -> (B, H, T, S)

((x @ w) * y).sum(dim=-1)  # (B, D) @ (D, E) -> (B, E), then row b against row b

x @ w  # (B, H, T, D) @ (H, D, E): w is broadcast as (1, H, D, E)
```

The `matmul` form of the scores needs the keys' last two axes swapped, because `@`
wants the summed axis second to last on the right. The form of `bilinear` shows why
it is not `x @ w @ y.T`: that last `@` pairs every row with every row. And in
`x @ w` the pairing of heads is done by broadcasting, aligned from the right:
`(B, H)` against `(H,)`. It is right because the head axis is the last of the leading
axes on both sides. A weight of shape `(B, D, E)`, one matrix per batch row, would
need `w[:, None]`.

The three forms add the same products in different orders, and their results differ
in the last bits. Over 300 random inputs in $[-2, 2]$ with sizes up to 8, in all three
frameworks, the largest difference from a float64 sum was `7e-6` (for `bilinear`,
whose results reach the tens), which is why the tests compare with a tolerance of
`1e-4` instead of exactly.

## Common mistakes

- **The result's letters in the wrong order.** `"bhtd,bhsd->bhst"` returns
  `(B, H, S, T)`: the scores with keys as rows. With a KV cache `T` and `S` differ
  and the next operation raises; in a test where they are equal, the softmax runs
  over the wrong axis.
- **Scaling by habit.** Dividing by $\sqrt{D}$ belongs to attention, and this
  function is specified as the dot products alone.
- **A new letter for an axis that should be paired.** `"bhtd,bgsd->bhts"` gives the
  head axis of `k` the letter `g`, which is not in the result, so it is summed: every
  query head is scored against the sum of all heads' keys. It runs for every input.
- **Every row against every row.** `(x @ w @ y.T).sum(dim=1)` builds the `(B, B)`
  table and sums each row of it. A batch of one cannot show the difference.
- **A bare `squeeze()`.** `(x[:, None, :] @ w @ y[:, :, None])` is `(B, 1, 1)`, and
  `squeeze()` removes the batch axis too when `B == 1`. Name the axes,
  `squeeze(-1).squeeze(-1)`, or index them away with `[:, 0, 0]`.
- **`w` transposed.** `"bd,ed,be->b"` and `"bhtd,hed->bhte"` read `w[e, d]`. With
  `D != E` they raise, and with a square `w` they multiply by its transpose. The
  transpose is easy to write by reflex after `torch.nn.Linear`, whose weight is
  stored as `(out, in)`.
- **One matrix for all heads.** `x @ w[0]` has the right shape for every input.
- **The weight indexed by position.** `"bhtd,tde->bhte"` runs when `H == T`.

`einsum` also broadcasts an axis of size 1 against a letter's size, in all three
frameworks. That is why some of the mistakes above run on inputs with `D == 1`,
`E == 1`, or `T == 1` and return a wrong shape or wrong values where an error would
be expected.

## Complexity

`attention_scores` does $O(B\,H\,T\,S\,D)$ multiplications, `bilinear`
$O(B\,D\,E)$, and `per_head_project` $O(B\,H\,T\,D\,E)$. The `(B, B)` table of the
row-sum mistake costs $O(B^2 E)$ on top.

## In the three frameworks

| | PyTorch | JAX | MLX |
|---|---|---|---|
| the operation | `torch.einsum(eq, *ops)` | `jnp.einsum(eq, *ops)` | `mx.einsum(eq, *ops)` |
| no `->` in the equation | result letters in alphabetical order | the same | the same |
| `...` for leading axes | yes | yes | yes |
| a repeated letter on one operand (`"ii->i"`) | the diagonal | the diagonal | the diagonal |
| an axis of size 1 against a larger one, same letter | broadcast | broadcast | broadcast |
| sizes that differ and are not 1 | `RuntimeError`; `ValueError` with three operands | `ValueError` | `ValueError` |
| integer operands, `"ij,jk->ik"` | int32 result | int32 result | `ValueError`: floating types only |
| integer operands, `"ij,ij->ij"` or `"ii->i"` | int32 result | int32 result | int32 result |
| swap the last two axes | `x.transpose(-2, -1)`, `x.mT` | `x.swapaxes(-2, -1)`, `x.mT` | `x.swapaxes(-2, -1)`, `x.mT` |
| `x.transpose(-2, -1)` on four axes | swaps the two | `TypeError`: wants all four axes | `ValueError`: wants all four axes |
| `x.transpose(0, 1, 3, 2)` | `TypeError`: use `x.permute(0, 1, 3, 2)` | permutes | permutes |
| `x.T` on four axes | reverses all axes, with a deprecation warning | reverses all axes | reverses all axes |
| `(B, H, T, D) @ (H, D, E)` | `(B, H, T, E)` | `(B, H, T, E)` | `(B, H, T, E)` |

Every entry was run on PyTorch 2.15, JAX 0.12, and MLX 0.32.4.

The equations are portable: all three functions of this problem are the same string
in every framework. What does not port is `transpose`. In PyTorch it swaps two axes;
in JAX and MLX it takes a full permutation, and the PyTorch call with two arguments
raises there. `swapaxes(-2, -1)` and `.mT` mean the same thing in all three. The
other difference is MLX's `einsum` on integers. An equation that sums a product of
two operands (`"ij,jk->ik"`, `"ij,j->i"`, `"i,i->"`) goes through `matmul`, which
accepts floating types only, and raises; cast such operands first. Equations with
no such sum (`"ij,ij->ij"`, `"ii->i"`, `"ij->ji"`, `"ij->"`) returned int32.
