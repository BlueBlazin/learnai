Let $m$ be the hidden activation vector, and write the second layer as a sum over hidden units:
$W_{\text{down}}\, m = \sum_j m_j \cdot (\text{column } j \text{ of } W_{\text{down}})$.
If a device holds only a subset of the hidden units $m_j$, which slice of $W_{\text{down}}$ does it need to compute its contribution to $y$, and which collective combines the contributions from all devices?

---

In `shard_weights`, compute `bounds = [i * h // n for i in range(n + 1)]` and slice between consecutive boundaries. Take rows of `w_up` and entries of `b_up` (the first layer's outputs), and take **columns** of `w_down` (the second layer's inputs). Keep `b_down` unsliced on every device. In `tp_mlp`, loop over the devices and compute `silu(x @ up.T + up_bias) @ down.T` on each device. Because SiLU is elementwise, each device can apply it locally without communicating.

---

```python
parts = [partial_output_of_device(i) for i in range(devs.n_devices)]
return devs.all_reduce(parts) + devs.get(0, "down_bias")
```

Run one `all_reduce` and add `down_bias` afterward, since adding it inside each device's partial output would include it `n` times. Avoid calling `all_gather` on the hidden slices, because that communicates the widest tensor in the layer to compute what each device can already multiply locally.
