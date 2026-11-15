The two rotation functions compute the same $D/2$ rotations at the same $D/2$
frequencies. Write one helper for the table of cosines and sines, shape `(T, D/2)`,
and ask of each layout only: which slice of `x` holds the first members $a$ of all
pairs, which holds the second members $b$, and where do $a'$ and $b'$ go back?

---

Interleaved: $a$ is `x[..., 0::2]`, $b$ is `x[..., 1::2]`; stack $a'$ and $b'$ on a
new last axis and reshape to put them side by side again. Half-split: $a$ is the
first half `x[..., :D//2]`, $b$ is the second half `x[..., D//2:]`; concatenate $a'$
and $b'$ along the last axis. In both, $a' = a\cos - b\sin$ and $b' = a\sin + b\cos$
with the same table.

For `permute_qk`, compare the two layouts as orderings of the same $(a_i, b_i)$:
interleaved is $a_0, b_0, a_1, b_1, \dots$ and half-split is
$a_0, a_1, \dots, b_0, b_1, \dots$. The weight's rows are the output features, so its
rows must be reordered the same way, separately inside each head.

---

Give the rows of `w` three indices, head, pair, and member, by reshaping
`(n_heads * d_head, d_in)` to `(n_heads, d_head // 2, 2, d_in)`: row
`h * d_head + 2 * i + c` is entry `[h, i, c]`. The half-split order is the same
entries listed with the member index before the pair index, so swap those two axes
and flatten back:

```python
w.reshape(n_heads, d_head // 2, 2, d_in).transpose(1, 2).reshape(n_heads * d_head, d_in)
```

Check it on one head of width 8, where the answer is rows `0, 2, 4, 6, 1, 3, 5, 7`.
Width 4 is a poor test: there the permutation `0, 2, 1, 3` is its own inverse, so the
conversion in the wrong direction looks right.
