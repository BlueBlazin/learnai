Consider two questions. First, if you multiply column $j$ of $W$ by $s_j$ and divide activation channel $j$ by $s_j$, the unquantized product $x W^\top$ does not change. Only quantization in between changes the result. Second, because the quantizer preserves each group's minimum and maximum and spaces the grid evenly between them, think about what happens to a column's effective rounding error when you scale it up inside its group, and what happens to its neighbors in the same group.

---

Loop over `grid` and compute four things per $\alpha$: the normalized scales, the scaled-quantized-unscaled weights, the mean squared output error, and the candidate list. Precompute the channel magnitudes `mean(abs(x_calib), axis=0)` and the unquantized target `x_calib @ w.T` once before the loop.

Write the group-wise quantizer as a helper that returns dequantized weights (`codes * step + lo`), handling constant groups with a `where` as in `groupwise-4bit`. Clamp the raw scales at `1e-4` *before* normalizing so that a dead channel with magnitude 0 does not cause division by zero.

---

Inside the loop over `grid`:

```python
s = maximum(act**alpha, 1e-4)
s = s / sqrt(max(s) * min(s))
w_hat = quantize_groups(w * s, bits, group_size) / s  # s broadcasts over columns
loss = mean((target - x_calib @ w_hat.T) ** 2)
```

Append each candidate's `s` and `loss` to lists. After the loop, `argmin` over the stacked losses returns the index of the first minimum, satisfying the tie-breaking rule. Return `candidates[best]` and `grid[best]`, not the loop variable `s` (which holds the last grid entry's scales).
