Write the second layer as a sum over hidden units:
$W_{\text{down}}\, m = \sum_j m_j \cdot (\text{column } j \text{ of } W_{\text{down}})$.
If a device holds only some of the hidden units $m_j$, which part of
$W_{\text{down}}$ does it need to turn them into a contribution to $y$, and what
operation combines the contributions of all devices?

---

`shard_weights`: compute the boundaries `bounds = [i * h // n for i in range(n + 1)]`
and slice between consecutive ones: rows of `w_up` and entries of `b_up` (the
outputs of the first layer), **columns** of `w_down` (the inputs of the second).
`b_down` is not sliced. In `tp_mlp`, loop over the devices; each computes
`silu(x @ up.T + up_bias) @ down.T` from its own four tensors. The activation is
elementwise, so a device can apply it to its slice without knowing the others.

---

```python
parts = [partial_output_of_device(i) for i in range(devs.n_devices)]
return devs.all_reduce(parts) + devs.get(0, "down_bias")
```

One all-reduce, and the bias after it: inside the sum it would be added `n` times.
Do not concatenate the hidden slices first (`all_gather`): that moves the widest
tensor of the layer to compute something each device could have computed locally.
