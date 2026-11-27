`mx.quantize` handles quantization and `mx.quantized_matmul` handles the matrix multiplication. Start by checking three details: which arrays `mx.quantize` returns in each mode, what exact attribute names the layer must store them under, and which layout parameters `mx.quantized_matmul` needs because they cannot be inferred from a packed `uint32` array.

---

An `nn.Module` acts like a dictionary. Assigning an MLX array to `self.<name>` registers it as a parameter named `<name>`, while assigning Python ints or strings stores ordinary attributes. Store `group_size`, `bits`, and `mode` as attributes, assign the outputs of `mx.quantize` to `self.weight`, `self.scales`, and (for affine mode) `self.biases`, and copy `linear.bias` when `"bias" in linear`. Avoid storing any other arrays on `self`. Starred unpacking handles both return counts cleanly. Writing `codes, scales, *biases = mx.quantize(...)` leaves `biases` empty for `"mxfp4"` and `"mxfp8"`.

---

The forward pass is a single call to `mx.quantized_matmul` with all layout parameters forwarded:

```python
y = mx.quantized_matmul(
    x,
    self.weight,
    scales=self.scales,
    biases=self.get("biases"),
    transpose=True,
    group_size=self.group_size,
    bits=self.bits,
    mode=self.mode,
)
if "bias" in self:
    y = y + self.bias
```

`self.get("biases")` returns `None` for modes without quantization biases. Passing `transpose=True` computes `x @ W_hat.T`, which matches a weight layout of `(out, in)`. Always pass `group_size`, `bits`, and `mode` explicitly so non-default configurations do not silently fall back to 4-bit groups of 64.
