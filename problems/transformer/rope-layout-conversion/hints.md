Both rotation functions apply the same $D/2$ rotations at the same $D/2$ frequencies. Start by writing one helper that computes the cosine and sine tables of shape `(T, D/2)`. Then, for each layout, ask which slice of `x` holds the first members $a$ of all pairs, which slice holds the second members $b$, and where $a'$ and $b'$ belong in the output.

---

For interleaved layout, $a$ is `x[..., 0::2]` and $b$ is `x[..., 1::2]`. Stack $a'$ and $b'$ on a new last axis and reshape back to `x.shape` to place them side by side again. For half-split layout, $a$ is `x[..., :D//2]` and $b$ is `x[..., D//2:]`, and you concatenate $a'$ and $b'$ along the last axis. Both layouts use $a' = a\cos - b\sin$ and $b' = a\sin + b\cos$ with the same cosine and sine tables.

For `permute_qk`, compare the two feature orderings of the same pairs $(a_i, b_i)$. Interleaved order is $a_0, b_0, a_1, b_1, \dots$ and half-split order is $a_0, a_1, \dots, b_0, b_1, \dots$. Because the rows of `w` produce the output features, its rows must be reordered in the exact same way inside each head.

---

Split the row axis of `w` into three indices (head, pair, and member) by reshaping `(n_heads * d_head, d_in)` to `(n_heads, d_head // 2, 2, d_in)`. Row `h * d_head + 2 * i + c` then lives at `[h, i, c]`. Half-split order places the member index before the pair index, so transpose those two axes and flatten back:

```python
w.reshape(n_heads, d_head // 2, 2, d_in).transpose(1, 2).reshape(n_heads * d_head, d_in)
```

Verify your permutation on a single head of width 8, where the resulting row order is `0, 2, 4, 6, 1, 3, 5, 7`. Width 4 is not a sufficient check because `0, 2, 1, 3` is its own inverse, so a permutation in the wrong direction still looks correct.
