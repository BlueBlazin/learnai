Write out what a weight in `(out, in)` layout does: output feature $i$ is the dot
product of the input with **row** $i$. If the fused projection's output is "the
features of projection 0, then the features of projection 1", where in the matrix do
the weights of projection 1 live?

---

In consecutive rows. `fuse` concatenates the weights along axis 0, and `split_fused`
cuts axis 0 into blocks of `parts[0]` rows, then `parts[1]` rows, and so on. Keep a
running offset: block $i$ is `w[start : start + parts[i]]`. Check what your
framework's split function does with a list before you use it: `torch.split` takes
sizes, while `torch.tensor_split`, `jnp.split`, `mx.split`, and `np.split` take the
positions at which to cut.

---

For `swiglu_fused`, do the one matrix product and cut its **output**, not the weight:

```python
z = x @ w_in.T  # (..., 2h)
gate, up = z[..., :h], z[..., h:]  # h = w_in.shape[0] // 2
y = (silu(gate) * up) @ w_down.T
```

The gate is the first half and is the only half that goes through SiLU. Cutting `z`
along its last axis is the same as cutting `w_in` along its first: both select the
same rows of the weight.
