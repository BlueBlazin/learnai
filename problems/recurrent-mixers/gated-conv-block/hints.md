Ask what output $t$ needs that is not present in the current call. It needs only the $L - 1$ preceding values of the convolution's input. Decide whether those values come from the block input $x$, the gated input $u = B \odot h$, or the convolution output $v$. That choice determines what `state` must store.

---

Concatenate `state` and the new rows of $u$ along the token axis to form `padded`, which has $L - 1 + T$ rows. Output $t$ is the weighted sum of rows $t, t + 1, \dots, t + L - 1$ of `padded`, where tap $j$ multiplies row $t + j$. You can compute this directly as a sum of $L$ shifted slices, each multiplied by one column of `w_conv`:

```python
conv = sum(padded[j : j + T] * w_conv[:, j] for j in range(L))
```

If you use a framework convolution function instead, pass `padded` (not `u`) with no extra padding and one group per channel. Framework convolutions do not flip the kernel either.

---

The new state is `padded[T:]`, which is the last $L - 1$ rows of the concatenated tensor. Slicing only from `u` fails whenever $T < L - 1$, including every single-token decoding step. When `state` is `None`, initialize it to $L - 1$ rows of zeros. Split `x @ w_in.T` into three contiguous blocks of columns to get `B`, `C`, and `h`, and return `(C * conv) @ w_out.T` alongside the new state.
