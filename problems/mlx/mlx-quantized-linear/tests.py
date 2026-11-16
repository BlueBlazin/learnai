"""Tests for mlx-quantized-linear.

Every case builds a float `nn.Linear`, hands it to the solution's `QLinear`, and checks:

1. the parameters: exactly the names of an MLX quantized checkpoint (`weight`,
   `scales`, `biases` for affine, `bias` if the float layer has one), with the values
   `mx.quantize` returns (computed here);
2. the output, against an oracle that does not use `mx.quantized_matmul`: for affine,
   NumPy unpacks the codes and multiplies by the dequantized matrix (the kit's
   `dequantize`, which equals the on-disk format of 8-bit MLX checkpoints); for the
   `mx` floating-point modes, `mx.dequantize` and an ordinary matmul;
3. how the output is computed: the lazy graph of `layer(x)` must contain a
   `QuantizedMatmul` and no `Matmul`, `Quantize`, or `Dequantize` operation
   (`mx.export_to_dot` lists the operations still pending behind an array). This runs
   on every machine;
4. on the Apple GPU: what the layer holds after the float layer is deleted, and the
   peak memory of a forward pass, on a 2048 x 2048 layer whose float32 weight is 16 MiB.
"""

import io
import re

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from learnai.judge import case, check
from learnai.kit import dequantize, mlx_memory
from mlx import nn

MIB = 2**20


def float_linear(weight, bias=None):
    """An evaluated nn.Linear holding the given NumPy weight (out, in) and bias."""
    n_out, n_in = weight.shape
    linear = nn.Linear(n_in, n_out, bias=bias is not None)
    linear.weight = mx.array(np.asarray(weight, dtype=np.float32))
    if bias is not None:
        linear.bias = mx.array(np.asarray(bias, dtype=np.float32))
    mx.eval(linear.parameters())
    return linear


def random_linear(ctx, n_in, n_out, bias=True):
    weight = ctx.randn(n_out, n_in, scale=1.0 / np.sqrt(n_in))
    # Shift half of the rows so that groups do not straddle zero the same way everywhere.
    weight[::2] += 0.5 / np.sqrt(n_in)
    return float_linear(weight, ctx.randn(n_out) if bias else None)


def parameters(layer):
    return dict(tree_flatten(layer.parameters()))


def graph_ops(*arrays):
    """Names of the operations still pending behind `arrays` (empty: evaluated)."""
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return re.findall(r'label ="(\w+)"', dot.getvalue())


def build(ctx, linear, group_size, bits, mode="affine"):
    """Construct the solution's layer and check that the float layer is untouched."""
    weight_before = linear.weight
    layer = ctx.solution.QLinear(linear, group_size, bits, mode)
    check.that(isinstance(layer, nn.Module), "QLinear: must be an nn.Module")
    check.that(
        isinstance(linear, nn.Linear) and linear.weight is weight_before,
        "QLinear: the float layer passed to the constructor was modified",
    )
    return layer


def check_parameters(layer, linear, group_size, bits, mode="affine"):
    """The parameter names, dtypes, shapes, and values of an MLX quantized checkpoint."""
    want = dict(
        zip(("weight", "scales", "biases"), mx.quantize(linear.weight, group_size, bits, mode=mode))
    )
    if "bias" in linear:
        want["bias"] = linear.bias
    got = parameters(layer)
    check.that(
        sorted(got) == sorted(want),
        f"QLinear: the parameters are {sorted(got)}, which are not the expected names",
        f"expected {sorted(want)} for mode {mode!r}"
        + (" and a float layer with a bias" if "bias" in linear else " and no bias"),
    )
    for name, expected in want.items():
        value = got[name]
        check.that(
            value.dtype == expected.dtype,
            f"QLinear: parameter {name} has dtype {value.dtype}, which is not the expected dtype",
            f"expected {expected.dtype}",
        )
        label = f"parameter {name}"
        if mx.issubdtype(expected.dtype, mx.floating):
            check.allclose(np.array(value), np.array(expected), label=label)
        else:
            check.shape(np.array(value), expected.shape, label=label)
            check.equal(np.array(value), np.array(expected), label=label)
    return got


def expected_output(linear, x, group_size, bits, mode="affine"):
    """x @ W_hat.T + bias without mx.quantized_matmul."""
    packed, scales, *biases = mx.quantize(linear.weight, group_size, bits, mode=mode)
    if mode == "affine":
        w_hat = dequantize(
            np.array(packed), np.array(scales), np.array(biases[0]), group_size, bits
        )
    else:
        w_hat = mx.dequantize(packed, scales, None, group_size, bits, mode=mode)
        w_hat = np.array(w_hat.astype(mx.float32))
    y = x.astype(np.float64) @ w_hat.astype(np.float64).T
    if "bias" in linear:
        y = y + np.array(linear.bias).astype(np.float64)
    return y.astype(np.float32)


def check_forward(layer, linear, x, group_size, bits, mode="affine", label="output"):
    out = layer(mx.array(x))
    check.that(isinstance(out, mx.array), "QLinear: calling the layer must return an mx.array")
    check.that(
        out.dtype == mx.float32,
        f"{label}: got dtype {out.dtype}, which is not the expected dtype",
        "expected float32",
    )
    want = expected_output(linear, x, group_size, bits, mode)
    check.allclose(np.array(out), want, rtol=1e-4, atol=1e-4, label=label)


def check_graph(layer, x):
    """The forward pass is one multiplication from the packed codes."""
    mx.eval(layer.parameters())
    ops = graph_ops(layer(x))
    detail = f"operations in the graph of layer(x): {sorted(set(ops))}"
    # MLX names the operation behind mx.dequantize "Quantize" in an exported graph.
    for banned, what in (
        ("Matmul", "an ordinary Matmul"),
        ("Dequantize", "a Dequantize"),
        ("Quantize", "a Quantize or Dequantize"),
    ):
        check.that(
            banned not in ops,
            f"QLinear: the forward pass contains {what} operation; it must multiply "
            "from the packed codes with mx.quantized_matmul",
            detail,
        )
    check.that(
        ops.count("QuantizedMatmul") == 1,
        f"QLinear: the forward pass contains {ops.count('QuantizedMatmul')} "
        "QuantizedMatmul operations instead of exactly one",
        detail,
    )


def run(ctx, linear, x, group_size, bits, mode="affine"):
    layer = build(ctx, linear, group_size, bits, mode)
    check_parameters(layer, linear, group_size, bits, mode)
    check_forward(layer, linear, x, group_size, bits, mode)
    return layer


def example_layer():
    """The layer of the statement: 64 inputs, 2 outputs, one group per row."""
    j = np.arange(64)
    weight = np.stack([1.0 - 0.1 * (j % 16), 2.9 - 0.06 * (j % 16)]).astype(np.float32)
    return float_linear(weight, [0.5, -0.2])


@case("the worked example: 64 inputs, 2 outputs, 4 bits, one group per row", public=True)
def worked_example(ctx):
    linear = example_layer()
    x = np.zeros(64, dtype=np.float32)
    x[[0, 1, 2]] = 1.0
    x[17] = 2.0
    layer = run(ctx, linear, x, 64, 4)
    got = parameters(layer)
    check.equal(np.array(got["weight"])[0, :2], np.array([0x76543210, 0xFEDCBA98]), label="weight")
    check.allclose(np.array(got["scales"]), [[-0.1], [-0.0604167]], label="scales")
    check.allclose(np.array(got["biases"]), [[1.0], [2.9]], label="biases")
    check.allclose(np.array(layer(mx.array(x))), [5.0, 13.997917], atol=1e-4, label="output")
    check.at_most("parameter_bytes", sum(v.nbytes for v in got.values()), 88, unit="bytes")


@case("random layers with and without a bias, batched inputs", public=True)
def random_layers(ctx):
    # Rejects bias_dropped (the first layer) and scales_biases_swapped.
    x = ctx.randn(3, 5, 128)
    with_bias = random_linear(ctx, 128, 48, bias=True)
    with_bias.bias = with_bias.bias + 3.0
    mx.eval(with_bias.parameters())
    run(ctx, with_bias, x, 64, 4)
    run(ctx, random_linear(ctx, 128, 48, bias=False), x, 64, 4)
    # A square layer: `x @ W` and `x @ W.T` have the same shape and different values.
    run(ctx, random_linear(ctx, 64, 64, bias=False), ctx.randn(7, 64), 64, 4)


@case("the forward pass multiplies from the packed codes: one QuantizedMatmul", public=True)
def multiplies_from_codes(ctx):
    # Rejects stores_dequantized and dequantizes_every_call on every machine.
    linear = random_linear(ctx, 128, 32)
    x = mx.array(ctx.randn(4, 128))
    mx.eval(x)
    check_graph(build(ctx, linear, 64, 4), x)
    check_graph(build(ctx, linear, 32, 8), x)


@case('mode="mxfp4": mx.quantize returns two arrays and there are no biases', public=True)
def mxfp4_mode(ctx):
    # Rejects ignores_mode. Group size 32 and 4 bits are the only choice for mxfp4.
    x = ctx.randn(6, 96)
    run(ctx, random_linear(ctx, 96, 40, bias=True), x, 32, 4, "mxfp4")
    layer = run(ctx, random_linear(ctx, 96, 40, bias=False), x, 32, 4, "mxfp4")
    check_graph(layer, mx.array(x))


# Hidden cases.


@case("other bit widths and group sizes: 8 bits in groups of 32, 2 bits in groups of 128")
def other_widths(ctx):
    # Rejects default_group_and_bits: the multiplication must be told the layout.
    x = ctx.randn(2, 256)
    for group_size, bits in [(32, 8), (128, 2), (32, 4), (64, 8), (128, 4), (64, 2)]:
        run(ctx, random_linear(ctx, 256, 24), x, group_size, bits)


@case("a square layer: 128 inputs and 128 outputs, 8 bits")
def square_layer(ctx):
    # Rejects transpose_false with a wrong answer and not with a shape error.
    linear = random_linear(ctx, 128, 128)
    run(ctx, linear, ctx.randn(128), 64, 8)
    run(ctx, linear, ctx.randn(2, 3, 128), 32, 4)


@case("the bias is the float layer's bias, exactly, and is absent when the layer has none")
def bias_handling(ctx):
    weight = ctx.randn(16, 64, scale=0.1)
    bias = np.linspace(-50.0, 50.0, 16, dtype=np.float32)
    linear = float_linear(weight, bias)
    layer = run(ctx, linear, ctx.randn(4, 64), 64, 4)
    check.equal(np.array(parameters(layer)["bias"]), bias, label="parameter bias")
    layer = run(ctx, float_linear(weight), ctx.randn(4, 64), 64, 4)
    check.that("bias" not in parameters(layer), "QLinear: a bias appeared for a layer without one")
    # Inputs of zero return the bias alone.
    layer = build(ctx, linear, 64, 4)
    check.allclose(np.array(layer(mx.zeros((64,)))), bias, label="output for a zero input")


@case('mode="mxfp8" in groups of 32')
def mxfp8_mode(ctx):
    linear = random_linear(ctx, 64, 64)
    layer = run(ctx, linear, ctx.randn(5, 64), 32, 8, "mxfp8")
    check_graph(layer, mx.array(ctx.randn(5, 64)))


@case("constant rows and rows far from zero")
def awkward_rows(ctx):
    weight = ctx.randn(8, 128, scale=0.05)
    weight[0] = 0.7  # a constant group: any scale reproduces it
    weight[1] = 0.0
    weight[2] += 40.0  # a narrow range far from zero
    weight[3, :64] -= 9.0
    run(ctx, float_linear(weight, ctx.randn(8)), ctx.randn(3, 128), 64, 4)
    run(ctx, float_linear(weight), ctx.randn(3, 128), 32, 8)


@case("parameter bytes at 4 bits in groups of 64: at most 0.2 of the float32 layer")
def parameter_bytes(ctx):
    # Rejects stores_dequantized by its size alone: 1.0 of the float layer.
    linear = random_linear(ctx, 512, 256)
    layer = build(ctx, linear, 64, 4)
    check_parameters(layer, linear, 64, 4)
    float_bytes = sum(v.nbytes for v in parameters(linear).values())
    quantized = sum(v.nbytes for v in parameters(layer).values())
    check.at_most("parameter_bytes_ratio", round(quantized / float_bytes, 4), 0.2)
    # 8 bits in groups of 64 is 9 bits per weight.
    layer = build(ctx, linear, 64, 8)
    quantized = sum(v.nbytes for v in parameters(layer).values())
    check.at_most("parameter_bytes_ratio_8bit", round(quantized / float_bytes, 4), 0.3)


@case(
    "a 2048 x 2048 layer: the float weight can be freed, and a forward pass allocates "
    "no float matrix",
    requires="mlx-gpu",
)
def memory_of_a_large_layer(ctx):
    # The float32 weight is 16 MiB; codes, scales, and biases at 4 bits are 2.5 MiB.
    weight = ctx.randn(2048, 2048, scale=0.02)
    x = mx.array(ctx.randn(8, 2048))
    mx.eval(x)
    mx.synchronize()
    with mlx_memory() as held:
        linear = float_linear(weight)
        layer = ctx.solution.QLinear(linear, 64, 4)
        mx.eval(layer.parameters())
        del linear
    # Rejects stores_dequantized (16 MiB, or 18.5 MiB with the codes kept as well).
    check.memory(held, at_most_mib=5.0, what="active", metric="layer_memory")

    mx.eval(layer(x))  # allocate whatever the first call allocates once
    with mlx_memory() as forward:
        out = layer(x)
        mx.eval(out)
    # Rejects dequantizes_every_call: a 16 MiB temporary on every forward pass.
    check.memory(forward, at_most_mib=4.0, metric="forward_peak_memory")
    want = x.astype(mx.float32) @ mx.dequantize(*mx.quantize(mx.array(weight), 64, 4), 64, 4).T
    check.allclose(np.array(out), np.array(want), rtol=1e-3, atol=1e-3, label="output")
