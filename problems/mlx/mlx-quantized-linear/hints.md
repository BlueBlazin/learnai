`mx.quantize` already does the quantization and `mx.quantized_matmul` already does the
multiplication. What is left is bookkeeping: which arrays does the first return in
each mode, under which names must the layer keep them, and which of its arguments does
the second need to be told again because it cannot read them off a packed uint32
array?

---

An `nn.Module` is a dict: assigning an array to `self.<name>` makes it a parameter
called `<name>`, and assigning a Python int or string makes an ordinary attribute. So
keep `group_size`, `bits`, and `mode` as attributes, assign the outputs of
`mx.quantize` to `self.weight`, `self.scales`, and (affine only) `self.biases`, and
copy `linear.bias` if `"bias" in linear`. Do not assign anything else that is an
array. A starred assignment handles both modes: `codes, scales, *biases = mx.quantize(...)`
leaves `biases` empty for `"mxfp4"`.

---

The forward pass is one call, with everything about the layout passed along:

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

`self.get("biases")` is `None` when the mode has no biases. `transpose=True` computes
`x @ W_hat.T`, which is what a layer with a weight of shape `(out, in)` needs. Leave
out `group_size` or `bits` and MLX assumes 64 and 4, which is right only for the
default layer.
