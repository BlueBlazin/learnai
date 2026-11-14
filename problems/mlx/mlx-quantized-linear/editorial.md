## The idea

A quantized layer is three small arrays and one operation.

`mx.quantize(w, group_size, bits)` cuts each row of `w` into groups of `group_size`
consecutive inputs, gives every group a scale and a bias, and replaces each weight by
a `bits`-bit code, packed `32 / bits` to a uint32 word. The weight a code stands for
is `code * scale + bias`. `mx.quantized_matmul(x, codes, scales, biases, ...)`
computes `x @ W_hat.T` without ever forming `W_hat`: the Metal kernel walks a row of
packed words, unpacks each code with a shift and a mask, multiplies it by the input,
and applies the group's scale and bias as it accumulates, which is the identity of
`quantized-matmul` executed on the GPU.

So the layer has nothing to compute. It has three things to get right:

1. **Keep what the checkpoint keeps.** `weight` (the packed codes), `scales`,
   `biases`, and the float layer's `bias`. An `nn.Module` turns every array assigned
   to it into a parameter under the attribute's name, so these four assignments *are*
   the checkpoint format: `layer.save_weights(...)` writes the tensors that
   `mlx_lm.convert` writes, and the layer loads what it wrote.
2. **Tell the kernel the layout.** A `(out, in * bits / 32)` array of uint32 does not
   say how many bits a code has or how long a group is. `group_size`, `bits`, and
   `mode` are plain attributes of the module, passed on every call.
3. **Keep nothing else.** The float matrix is the thing the layer exists to get rid
   of.

## Reference solution (MLX)

```python
class QLinear(nn.Module):
    def __init__(self, linear, group_size=64, bits=4, mode="affine"):
        super().__init__()
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self.weight, self.scales, *biases = mx.quantize(linear.weight, group_size, bits, mode=mode)
        if biases:
            self.biases = biases[0]
        if "bias" in linear:
            self.bias = linear.bias
        self.freeze()

    def __call__(self, x):
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
        return y
```

The starred assignment covers both shapes of the result: three arrays for `"affine"`,
two for `"mxfp4"` and `"mxfp8"`. `self.get("biases")` is the module's `dict.get`, so
it is `None` exactly when no biases were stored. `freeze()` is not judged; it is what
`nn.QuantizedLinear` does, because packed integer codes cannot receive gradients and
an optimizer should not be handed them.

## What `mx.quantize` computes

Not quite the textbook of `groupwise-4bit`, where the bias is the group's minimum and
the scale is `(max - min) / (2^bits - 1)`. For each group MLX

1. takes the end of the range that is **farther from zero** as the starting point
   (the `bias`), and gives the scale the sign that walks from there toward the other
   end, so scales are negative about half of the time;
2. rounds `bias / scale` to an integer `q0` and replaces the scale by `bias / q0`.
   After that, zero is an exact multiple of the scale away from the bias: a weight of
   exactly `0.0` is recovered as exactly `0.0` whenever zero lies inside the group's
   range.

In the worked example, row 1 spans `[2.0, 2.9]`. The starting point is `2.9`, the
first scale is `-0.06`, `2.9 / -0.06 = -48.33` rounds to `-48`, and the final scale is
`2.9 / -48 = -0.0604167`. The price is visible: the far end `2.0` comes back as
`1.99375`, where a min/max grid would have returned it exactly. A constant group
needs no special case, because the scale is clamped away from zero before the
division.

The layer does not care. It stores what `mx.quantize` returns and hands it back to
`mx.quantized_matmul`, and any quantizer that writes the same three arrays (a
calibrated one, a learned one) runs through the same forward pass. That is the reason
to keep the two functions apart in your head: the format is `code * scale + bias`
per group; how the scales and biases were chosen is the quantizer's business.

The `mx` modes are the same picture with a different code. `"mxfp4"` stores each
weight as a 4-bit floating-point number (one sign bit, two exponent bits, one mantissa
bit) and each group of 32 shares one power-of-two scale, stored as a single byte. No
bias, and `scales` is uint8, which is why the layer must not assume three float
arrays.

## What it buys

Measured on this machine (M1 Max, MLX 0.32.4), one row of input, best of five:

| layer | float32 weight | 4-bit parameters | float matmul | `quantized_matmul` | dequantize, then matmul |
|---|---|---|---|---|---|
| 2048 × 2048 | 16 MiB | 2.5 MiB | 0.30 ms | 0.25 ms | 0.39 ms |
| 4096 × 4096 | 64 MiB | 10 MiB | 0.53 ms | 0.43 ms | 0.93 ms |

Multiplying from the codes is no slower than the float matmul (it reads a sixth of the
bytes), and with 64 rows of input it is twice as fast on the larger layer. The layer
that dequantizes on every call is the slowest of the three, and its peak memory for
one forward pass of the 2048 × 2048 layer is 16.1 MiB against 0.06 MiB. It has the
size of a float model and the accuracy of a quantized one.

## Common mistakes

- **Storing the dequantized weight.** `self.weight = mx.dequantize(...)` gives a layer
  that computes the right numbers from a float matrix. It is the float layer with
  quantization noise added: same bytes, same bandwidth, and a `weight` that no
  quantized checkpoint can be loaded into.
- **Dequantizing on every call.** The parameters are right and the forward pass
  rebuilds the float matrix each time: `x @ mx.dequantize(...).T`. Nothing in the
  output shows it. The graph does (a `Matmul` where a `QuantizedMatmul` should be),
  and so does the allocator.
- **`transpose=False`.** `mx.quantized_matmul` can multiply by `W_hat` or by its
  transpose. A layer's weight is `(out, in)`, so it is always the transpose; with
  `False`, a square layer silently computes `x @ W_hat` and every other layer raises
  a shape error.
- **Dropping the bias.** The float layer's `bias` (one per output) has nothing to do
  with the quantization `biases` (one per group). The first is copied as it is; the
  second comes from `mx.quantize`. Having two things with nearly the same name is the
  format's fault, and forgetting one of them is the usual result.
- **Swapping scales and biases.** `mx.quantize` returns `(codes, scales, biases)`, in
  that order. Unpacked the other way round, every group is dequantized as
  `code * bias + scale`: a model that loads, runs, and produces noise.
- **Not passing `group_size` and `bits` to the matmul.** The defaults (64 and 4) match
  the default layer, so the mistake survives until the first 8-bit model or the first
  recipe with groups of 32.
- **Ignoring `mode`.** Always quantizing in affine mode makes `"mxfp4"` a silent
  no-op: the layer works, with a different format from the one that was asked for,
  and a `biases` tensor that an mxfp4 checkpoint does not have.
- **Wrapping `nn.QuantizedLinear`.** That is the layer this problem builds; outside
  the exercise, it is of course the one to use.

## Complexity

Construction is one pass over the weight. The layer stores
`out * in * bits / 8` bytes of codes plus two float32 numbers per group:
`bits + 64 / group_size` bits per weight, so 5 for the default and 9 for 8 bits in
groups of 64. Real checkpoints keep scales and biases in 16-bit floats, which is where
the familiar 4.5 and 8.5 bits per weight come from. A forward pass reads the packed
codes once and allocates only its output.

## In the wild

`nn.QuantizedLinear` is this class, with `from_linear` as the constructor, and
`nn.Linear.to_quantized` calls it. `nn.quantize(model, ...)` walks a module tree and
swaps every `Linear` and `Embedding` for its quantized twin, which is the subject of
`mlx-quantize-module-tree`. The 8-bit checkpoint of the catalog's larger house model
stores, for every layer, exactly the three tensors of this problem: `weight` as U32
of shape `[out, in / 4]`, and `scales` and `biases` as BF16 of shape `[out, in / 64]`.
