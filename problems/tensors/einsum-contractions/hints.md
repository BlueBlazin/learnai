For each function, write out the summation formula from the specification and inspect its index letters. The letters on the left-hand side become the output axes after `->`, in that order. Any letter that appears only under the sum is omitted after `->`. Together, those letters give the full `einsum` equation.

---

- `attention_scores`: `q` has axes `b h t d`, `k` has axes `b h s d`, and the output has axes `b h t s`. Only `d` is summed over.
- `bilinear`: `x` has axes `b d`, `w` has axes `d e`, `y` has axes `b e`, and the output has axis `b`. `einsum` accepts all three operands in one call. Using the same `b` on `x` and `y` pairs row `b` with row `b`.
- `per_head_project`: `x` has axes `b h t d`, `w` has axes `h d e`, and the output has axes `b h t e`. Sharing `h` matches each head of `x` with its own matrix in `w`.

If you use `@` instead, swap `S` and `D` on `k` so `D` is contracted. For `bilinear`, `x @ w` has shape `(B, E)` and still needs a row-by-row dot product with `y`. For `per_head_project`, `(B, H, T, D) @ (H, D, E)` automatically broadcasts `(H,)` against `(B, H)`.

---

The key lines, in PyTorch (`jnp.einsum` and `mx.einsum` take the same equations):

```
torch.einsum("bhtd,bhsd->bhts", q, k)       # q @ k.transpose(-2, -1)

torch.einsum("bd,de,be->b", x, w, y)        # ((x @ w) * y).sum(dim=-1)

torch.einsum("bhtd,hde->bhte", x, w)        # x @ w
```
