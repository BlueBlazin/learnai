What does output $t$ need that is not in the current call? Only the $L - 1$ values of
$u$ that came just before. Which tensor do they belong to: the block's input $x$, the
gated product $u = B \odot h$, or the convolution's output? The answer decides what
the state holds.

---

Concatenate the state and the new rows of $u$ into `padded`, with $L - 1 + T$ rows.
Output $t$ is a weighted sum of rows $t, t + 1, \dots, t + L - 1$ of `padded`, tap $j$
on row $t + j$. That is a sum of $L$ shifted slices, each multiplied by one column of
`w_conv`:

```python
conv = sum(padded[j : j + T] * w_conv[:, j] for j in range(L))
```

With a library convolution, pass `padded` (not `u`), no extra padding, and one group
per channel; the library does not flip the kernel either.

---

The new state is `padded[T:]`, the last $L - 1$ rows of the concatenation. Slicing
the last rows of `u` alone is wrong whenever the call has fewer than $L - 1$ tokens,
which includes every one-token decoding step. For `state=None` use $L - 1$ rows of
zeros. The output is `(C * conv) @ w_out.T`, with `B, C, h` the three consecutive
blocks of `x @ w_in.T`.
