## The idea

`mx.quantize(w, group_size, bits)` splits each row of a float weight matrix `w` into groups of `group_size` consecutive inputs, computes a scale and bias for each group, and packs `32 / bits` quantized codes into each `uint32` word. Each code represents `code * scale + bias`. During inference, `mx.quantized_matmul(x, codes, scales, biases, ...)` computes `x @ W_hat.T` without materializing the dequantized matrix `W_hat`. Its Metal kernel walks each row of packed words, unpacks codes with shifts and masks, multiplies by the input, and applies the group scale and bias during accumulation (the same identity implemented in `quantized-matmul`, running on the GPU).

Because MLX provides both primitives, the module has three responsibilities:

1. **Store the exact checkpoint tensors.** Save `weight` (packed codes), `scales`, `biases` (for affine mode), and the original `bias` (if present). An `nn.Module` registers every array attribute as a parameter under its attribute name, so assigning these attributes makes `layer.save_weights(...)` and checkpoint loading compatible with `mlx_lm.convert`.
2. **Record the packing layout.** A `uint32` array of shape `(out, in * bits / 32)` does not encode its bit width, group size, or mode. Store `group_size`, `bits`, and `mode` as plain Python attributes and pass them to `mx.quantized_matmul` on every call.
3. **Discard the float weight.** Keep no reference to `linear` or `linear.weight`, so the float matrix can be freed immediately.

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

Starred unpacking handles both return counts from `mx.quantize`: three arrays for `"affine"` and two for `"mxfp4"` and `"mxfp8"`. Because `nn.Module` inherits from `dict`, `self.get("biases")` returns `None` whenever no quantization biases were stored. Calling `self.freeze()` is not required by the tests, but matches `nn.QuantizedLinear` so optimizers do not attempt to update packed integer codes.

## What `mx.quantize` computes

MLX's affine quantizer differs slightly from the basic min/max scheme in `groupwise-4bit` (where `bias = min` and `scale = (max - min) / (2^bits - 1)`). For each group, MLX does two things:

1. It chooses the range endpoint **farther from zero** as `bias`, and gives `scale` the sign that steps toward the other endpoint. As a result, scales are negative roughly half the time.
2. It rounds `bias / scale` to an integer `q0` and replaces `scale` with `bias / q0`. This ensures zero is an exact multiple of `scale` away from `bias`, so an exact `0.0` weight is reconstructed as `0.0` whenever zero lies inside the group's range.

In the worked example, row 1 spans `[2.0, 2.9]`. The starting point is `2.9`, the initial step is `-0.06`, and `2.9 / -0.06 = -48.33` rounds to `-48`. The adjusted scale is `2.9 / -48 = -0.0604167`. The trade-off appears at the far end: `2.0` reconstructs as `1.99375`, whereas a plain min/max grid would hit `2.0` exactly. Constant groups need no special case because the scale is clamped away from zero before division.

`QLinear` itself does not depend on how the scales and biases were chosen. Any quantizer (calibrated or learned) that writes `weight`, `scales`, and `biases` in the `code * scale + bias` format works with the same `mx.quantized_matmul` forward pass.

The microscaling modes follow the same structure with floating-point codes. `"mxfp4"` stores each weight as a 4-bit float (1 sign bit, 2 exponent bits, 1 mantissa bit) with one shared power-of-two scale per group of 32 stored in a `uint8`. There is no bias tensor and `scales` is `uint8`, which is why the layer cannot assume three float arrays.

## What it buys

Measured on an M1 Max (MLX 0.32.4) for a single input row (best of five runs):

| layer | float32 weight | 4-bit parameters | float matmul | `quantized_matmul` | dequantize, then matmul |
|---|---|---|---|---|---|
| 2048 × 2048 | 16 MiB | 2.5 MiB | 0.30 ms | 0.25 ms | 0.39 ms |
| 4096 × 4096 | 64 MiB | 10 MiB | 0.53 ms | 0.43 ms | 0.93 ms |

Multiplying directly from packed codes is faster than the float matmul because it reads one-sixth as many bytes, and on 64 input rows it runs twice as fast on the larger layer. Dequantizing on every call is the slowest option and peaks at 16.1 MiB for one forward pass on the 2048 × 2048 layer (compared to 0.06 MiB for `quantized_matmul`), paying float memory costs for quantized accuracy.

## Common mistakes

- **Storing the dequantized weight.** Setting `self.weight = mx.dequantize(...)` produces the right outputs, but keeps a full float matrix with the same memory and bandwidth cost as `nn.Linear` and breaks checkpoint compatibility.
- **Dequantizing inside `__call__`.** Computing `x @ mx.dequantize(...).T` matches numerically, but puts a `Matmul` and `Dequantize` in the graph and allocates the full float weight on every call.
- **Using `transpose=False`.** `mx.quantized_matmul` supports multiplying by either `W_hat` or its transpose. Because linear weights have shape `(out, in)`, you must pass `transpose=True`. With `False`, non-square layers raise a shape error and square layers silently compute `x @ W_hat`.
- **Confusing `bias` and `biases`.** `linear.bias` is the optional per-output additive bias of the linear layer, while `biases` is the per-group affine quantization offset from `mx.quantize`. Dropping either one causes silent numerical errors.
- **Swapping `scales` and `biases`.** `mx.quantize` returns `(codes, scales, biases)` in that order. Swapping the last two evaluates `code * bias + scale` and produces garbage outputs without raising an error.
- **Omitting `group_size` or `bits` in `mx.quantized_matmul`.** The defaults (64 and 4) hide the bug on default layers and fail as soon as a test uses 8-bit weights or `group_size=32`.
- **Ignoring `mode`.** Hardcoding affine quantization turns `"mxfp4"` into a silent no-op and attaches an unexpected `biases` parameter that mxfp4 checkpoints do not have.
- **Wrapping `nn.QuantizedLinear`.** That is the built-in class this exercise rebuilds (and is forbidden here), though it is the right class to use in production code.

## Complexity

Construction makes a single pass over the weight matrix. The layer stores `out * in * bits / 8` bytes for packed codes plus two float32 values per group, totaling `bits + 64 / group_size` bits per weight (5 bits for 4-bit/64-group, 9 bits for 8-bit/64-group). Real checkpoints store scales and biases in 16-bit floats, yielding 4.5 and 8.5 bits per weight. Each forward pass reads the packed codes once and allocates only the output tensor.

## In the wild

`nn.QuantizedLinear` implements this exact design, with `from_linear` as its constructor (called by `nn.Linear.to_quantized`). `nn.quantize(model, ...)` walks a module tree and replaces every `Linear` and `Embedding` layer with its quantized counterpart (covered in `mlx-quantize-module-tree`). In practice, 8-bit MLX checkpoints store these exact three tensors per layer: `weight` as `uint32` of shape `[out, in / 4]`, and `scales` and `biases` as `bfloat16` of shape `[out, in / 64]`.
