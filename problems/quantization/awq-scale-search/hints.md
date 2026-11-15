Two questions. First: if you multiply column $j$ of $W$ by $s_j$ and divide
activation $j$ by $s_j$, what changes in $x W^\top$? Nothing, until you quantize in
between. Second: the quantizer keeps the *group's* extremes exactly and rounds the
rest to a grid whose step is set by those extremes. What happens to a column's
relative error when you scale it up inside its group, and what happens to its
neighbors?

---

Structure the function as a loop over `grid` around four lines: the scales for this
$\alpha$, the scaled-quantized-unscaled weight, the output error, and bookkeeping.
Everything outside the loop is computed once: the channel magnitudes
`mean(abs(x_calib), axis=0)` and the target output `x_calib @ w.T`.

Write the quantizer as a helper that returns dequantized weights (`codes * step +
lo`), with the constant-group case handled by a `where` as in `groupwise-4bit`.
Remember the clamp at `1e-4` *before* normalizing: a dead channel has magnitude 0,
and $0^\alpha = 0$ would later be divided by.

---

Inside the loop:

```python
s = maximum(act**alpha, 1e-4)
s = s / sqrt(max(s) * min(s))
w_hat = quantize_groups(w * s, bits, group_size) / s  # s broadcasts over columns
loss = mean((target - x_calib @ w_hat.T) ** 2)
```

Keep every candidate's `s` and loss in two lists. After the loop, `argmin` over the
stacked losses returns the first minimum, which is the tie rule; return
`candidates[best]` and `grid[best]`. Do not return the loop variable `s`: after the
loop it holds the scales of the *last* grid entry.
