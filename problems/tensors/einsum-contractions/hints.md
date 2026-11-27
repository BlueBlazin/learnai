For each function, copy the formula from the specification and list its letters.
Which letters appear on the left-hand side (they are the result's axes, in that
order)? Which appear only under the sum? That list is the whole `einsum` equation.

---

- `attention_scores`: `q` is `b h t d`, `k` is `b h s d`, the result is `b h t s`.
  `d` is the only letter that is summed.
- `bilinear`: `x` is `b d`, `w` is `d e`, `y` is `b e`, the result is `b`. Three
  operands in one equation are fine. The `b` on `x` and the `b` on `y` are the same
  letter, which is what pairs row `b` with row `b`.
- `per_head_project`: `x` is `b h t d`, `w` is `h d e`, the result is `b h t e`. The
  `h` on `w` is the `h` on `x`.

With `@` instead: the keys need `S` and `D` swapped so that `D` is summed; `x @ w`
is `(B, E)` and still has to meet `y` row by row; and `(B, H, T, D) @ (H, D, E)`
broadcasts the `(H,)` of `w` against the `(B, H)` of `x`.

---

The key lines, in PyTorch (`jnp.einsum` and `mx.einsum` take the same equations):

```
torch.einsum("bhtd,bhsd->bhts", q, k)       # q @ k.transpose(-2, -1)

torch.einsum("bd,de,be->b", x, w, y)        # ((x @ w) * y).sum(dim=-1)

torch.einsum("bhtd,hde->bhte", x, w)        # x @ w
```
