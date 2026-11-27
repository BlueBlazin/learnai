In `(out, in)` layout, output feature $i$ is the dot product of the input with **row** $i$ of the weight matrix. If a fused projection outputs the features of projection 0 followed by the features of projection 1, where in the weight matrix do the weights of projection 1 sit?

---

They sit in consecutive rows. `fuse` concatenates the weights along axis 0, and `split_fused` slices axis 0 into blocks of `parts[0]` rows, then `parts[1]` rows, and so on. You can track a running offset so block $i$ is `w[start : start + parts[i]]`. Before using a framework split helper, check how it interprets a list argument. `torch.split` takes section sizes, whereas `torch.tensor_split`, `jnp.split`, `mx.split`, and `np.split` take cut positions.

---

For `swiglu_fused`, run one matrix product and split its **output** rather than the weight:

```python
z = x @ w_in.T  # (..., 2h)
gate, up = z[..., :h], z[..., h:]  # h = w_in.shape[0] // 2
y = (silu(gate) * up) @ w_down.T
```

The gate is the first half and is the only half passed through SiLU. Splitting `z` along its last axis selects the exact same features as splitting `w_in` along axis 0.
