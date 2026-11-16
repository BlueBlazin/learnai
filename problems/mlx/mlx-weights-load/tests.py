"""Tests for mlx-weights-load.

Every case writes a small safetensors file with `mx.save_safetensors`, calls the
solution directly (so that `EvalSpy` and the graph probes see what it did), and checks:

1. the loaded parameters, exactly, against the numbers written (bfloat16 and float16
   round trips are exact), with the dtype of the module unchanged, and the model's
   output against a model built here from the same numbers;
2. the return value and the `ValueError`s the statement specifies, and that a failed
   load leaves the module as it was;
3. evaluation: the parameters come back evaluated (`mx.export_to_dot` lists no pending
   operation behind them), nothing was read implicitly, and a tensor the module does
   not use is never read from the file. The last is checked twice: `mx.load` is
   wrapped for the duration of the call so the test holds the arrays it returned, and
   an unexpected tensor must still be a pending `Load` afterwards (exact, on every
   machine); and, on the GPU, a 32 MiB unexpected tensor must not show in peak memory.
"""

import contextlib
import io
import os
import shutil
import tempfile

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from learnai.judge import case, check
from learnai.kit import EvalSpy, MLXTinyLM, mlx_memory
from mlx import nn

FN = "load_into"
TOKENS = [0, 17, 30, 5, 9, 23]  # all below 32, the smallest vocabulary used here


class Net(nn.Module):
    """The module of the worked example: an embedding and a Linear layer with a bias."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 2)
        self.out = nn.Linear(2, 3)

    def __call__(self, ids):
        return self.out(self.embed(ids))


class Nested(nn.Module):
    """Linear layers in a list, in a dict, and at the top level, with and without bias."""

    def __init__(self):
        super().__init__()
        self.blocks = [nn.Linear(4, 6), nn.Linear(6, 6, bias=False)]
        self.heads = {"a": nn.Linear(6, 3), "b": nn.Linear(6, 2, bias=False)}
        self.scale = mx.ones((6,))
        self.norm = nn.LayerNorm(6)

    def __call__(self, x):
        h = self.blocks[1](nn.relu(self.blocks[0](x))) * self.scale
        h = self.norm(h)
        return self.heads["a"](h), self.heads["b"](h)


EXAMPLE = {
    "embed.weight": np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]], dtype=np.float32
    ),
    "out.weight": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    "out.bias": np.array([0.5, 0.0, -0.5], dtype=np.float32),
}


def pending_ops(tree):
    """Number of operations not yet computed behind the arrays in `tree` (0: evaluated)."""
    arrays = [leaf for _, leaf in tree_flatten([tree]) if isinstance(leaf, mx.array)]
    if not arrays:
        return 0
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return dot.getvalue().count("shape=rectangle")


@contextlib.contextmanager
def watch_load():
    """Collect the arrays `mx.load` returns while the block runs: {name: array}."""
    seen = {}
    original = mx.load

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        tensors = result[0] if isinstance(result, tuple) else result
        if isinstance(tensors, dict):
            seen.update(tensors)
        return result

    mx.load = load
    try:
        yield seen
    finally:
        mx.load = original


def write(ctx, tensors):
    """Write {name: mx.array} to a fresh safetensors file; removed after the case."""
    folder = tempfile.mkdtemp(prefix="learnai-weights-")
    ctx.defer(lambda: shutil.rmtree(folder, ignore_errors=True))
    path = os.path.join(folder, "model.safetensors")
    mx.save_safetensors(path, tensors)
    return path


def snapshot(module):
    """The module's parameters as numpy arrays (float32 or float16 as they are)."""
    return {
        name: (np.array(value.astype(mx.float32)), value.dtype)
        for name, value in tree_flatten(module.parameters())
    }


def source_values(ctx, module):
    """Random float32 values for every parameter of `module`, at the scale it has now."""
    values = {}
    for name, value in tree_flatten(module.parameters()):
        current = np.array(value.astype(mx.float32))
        if current.ndim == 1:  # norm gains, biases, scales: around 1, never exactly 1
            values[name] = ctx.uniform(*current.shape, low=0.5, high=1.5)
        else:
            spread = float(current.std()) or 1.0
            values[name] = ctx.randn(*current.shape, scale=spread)
    return values


def as_file(values, *, dtype=mx.float32, prefix="", transposed=()):
    """Tensors for `write`: cast to `dtype`, renamed, the names in `transposed` as (in, out)."""
    tensors = {}
    for name, value in values.items():
        array = mx.array(value)
        if name in transposed:
            array = array.T
        tensors[prefix + name] = array.astype(dtype)
    return tensors


def through(values, dtype):
    """What `values` are after a round trip through `dtype` (bfloat16 rounds)."""
    return {
        name: np.array(mx.array(value).astype(dtype).astype(mx.float32))
        for name, value in values.items()
    }


def load(ctx, module, path, **options):
    """Call the solution; check the form of what it returns and how it evaluated."""
    with watch_load() as seen, EvalSpy() as spy:
        result = ctx.solution.load_into(module, path, **options)
    pending = pending_ops(module.parameters())
    # Measured before anything below can evaluate them: the tensors of the file that
    # were read although no parameter took them.
    current = {prefix_free for prefix_free, _ in tree_flatten(module.parameters())}
    prefix = options.get("prefix", "")
    read_unused = sorted(
        name
        for name, array in seen.items()
        if not (name.startswith(prefix) and name[len(prefix) :] in current)
        and pending_ops(array) == 0
    )
    check.that(
        spy.implicit == 0,
        f"{FN}: {spy.implicit} implicit evaluations (.item(), .tolist(), np.array, print, "
        "...); cast and transpose in MLX and evaluate with mx.eval",
        spy.describe(),
    )
    check.that(
        pending == 0,
        f"{FN}: returned with parameters that are not evaluated ({pending} operations pending)",
        "evaluate the module's parameters before returning",
    )
    check.that(
        not read_unused,
        f"{FN}: evaluated {len(read_unused)} tensor(s) of the file that the module does not use",
        f"read from disk although unexpected: {read_unused}",
    )
    check.that(
        isinstance(result, tuple)
        and len(result) == 2
        and all(isinstance(part, list) for part in result)
        and all(isinstance(name, str) for part in result for name in part),
        f"{FN}: must return a tuple (missing, unexpected) of two lists of names",
        f"got {result!r}",
    )
    return result


def must_raise(ctx, module, path, **options):
    """The call raises ValueError, as the statement specifies for this file."""
    check.raises(ValueError, lambda: ctx.solution.load_into(module, path, **options), label=FN)


def check_parameters(module, expected, before, *, label="parameters"):
    """Every parameter has the dtype it had before and the expected numbers, exactly."""
    after = snapshot(module)
    check.that(
        set(after) == set(before),
        f"{FN}: the module's parameter tree changed (parameters were added or removed)",
        f"before: {sorted(before)}\nafter: {sorted(after)}",
    )
    for name, (got, dtype) in after.items():
        was = before[name][1]
        check.that(
            dtype == was,
            f"{FN}: {name} has dtype {dtype} after loading, which is not the dtype it had",
            f"the parameter was {was} before loading; cast each tensor to it",
        )
        want = expected[name]
        if dtype == mx.float16:
            want = want.astype(np.float16).astype(np.float32)
        check.that(
            got.shape == want.shape,
            f"{FN}: {name} has shape {got.shape} after loading, which is not its shape",
            f"expected {want.shape}",
        )
        check.allclose(got, want, rtol=0.0, atol=0.0, label=f"{label}: {name}")


def unchanged(module, before):
    check_parameters(module, {k: v[0] for k, v in before.items()}, before, label="untouched")


def model_from(values, **options):
    """An MLXTinyLM holding exactly `values`: the oracle for the forward pass."""
    lm = MLXTinyLM(0, **options)
    lm.update(tree_unflatten([(name, mx.array(value)) for name, value in values.items()]))
    return lm


def check_logits(module, values, **options):
    got, _ = module(mx.array(TOKENS))
    want, _ = model_from(values, **options)(mx.array(TOKENS))
    check.allclose(
        np.array(got.astype(mx.float32)), np.array(want), rtol=1e-4, atol=1e-5, label="logits"
    )


def linear_names(module):
    return {
        f"{name}.weight" for name, child in module.named_modules() if isinstance(child, nn.Linear)
    }


@case("the worked example: a bfloat16 embedding and a float32 layer, prefix 'model.'", public=True)
def worked_example(ctx):
    net = Net()
    before = snapshot(net)
    tensors = as_file(EXAMPLE, prefix="model.")
    tensors["model.embed.weight"] = tensors["model.embed.weight"].astype(mx.bfloat16)
    path = write(ctx, tensors)
    missing, unexpected = load(ctx, net, path, prefix="model.")
    check.equal(missing, [], label="missing")
    check.equal(unexpected, [], label="unexpected")
    check_parameters(net, EXAMPLE, before)
    out = np.array(net(mx.array([1, 4])))
    check.allclose(
        out, np.array([[1.5, 3.0, 4.5], [0.5, 2.0, 3.5]], np.float32), label="net([1, 4])"
    )


@case("the worked example, second file: a missing bias and a tensor nobody wants", public=True)
def strict_and_not(ctx):
    net = Net()
    before = snapshot(net)
    tensors = as_file(EXAMPLE, prefix="model.")
    del tensors["model.out.bias"]
    tensors["model.vision.weight"] = mx.ones((2, 2))
    path = write(ctx, tensors)

    must_raise(ctx, net, path, prefix="model.")
    unchanged(net, before)  # a failed load must not have touched the module

    missing, unexpected = load(ctx, net, path, prefix="model.", strict=False)
    check.equal(missing, ["out.bias"], label="missing")
    check.equal(unexpected, ["model.vision.weight"], label="unexpected")
    expected = dict(EXAMPLE, **{"out.bias": before["out.bias"][0]})
    check_parameters(net, expected, before)


@case("a whole transformer: every parameter, then the logits", public=True)
def whole_model(ctx):
    lm = MLXTinyLM(2)
    before = snapshot(lm)
    values = source_values(ctx, lm)
    path = write(ctx, as_file(values))
    check.equal(load(ctx, lm, path), ([], []), label="(missing, unexpected)")
    check_parameters(lm, values, before)
    check_logits(lm, values)


@case("transpose_linear: Linear weights stored as (in, out), the embedding as it is", public=True)
def transposed_linear(ctx):
    options = {"n_kv_heads": 2}
    lm = MLXTinyLM(3, **options)
    before = snapshot(lm)
    values = source_values(ctx, lm)
    path = write(ctx, as_file(values, transposed=linear_names(lm)))
    check.equal(load(ctx, lm, path, transpose_linear=True), ([], []), label="result")
    check_parameters(lm, values, before)
    check_logits(lm, values, **options)


# Hidden cases. Each names the mutant it is written for.


@case("a float16 module keeps float16, from bfloat16 and from float32 files")
def half_precision_module(ctx):
    # Rejects keeps_checkpoint_dtype: the parameters would become bfloat16 or float32.
    for file_dtype in (mx.bfloat16, mx.float32):
        lm = MLXTinyLM(4, dtype=mx.float16)
        before = snapshot(lm)
        values = source_values(ctx, lm)
        path = write(ctx, as_file(values, dtype=file_dtype, prefix="model."))
        check.equal(load(ctx, lm, path, prefix="model."), ([], []), label="result")
        check_parameters(lm, through(values, file_dtype), before)


@case("a bfloat16 checkpoint into a float32 model: exact values, float32 logits")
def bf16_into_float32(ctx):
    lm = MLXTinyLM(5)
    before = snapshot(lm)
    values = source_values(ctx, lm)
    path = write(ctx, as_file(values, dtype=mx.bfloat16))
    load(ctx, lm, path)
    rounded = through(values, mx.bfloat16)
    check_parameters(lm, rounded, before)
    check_logits(lm, rounded)


@case("a tensor of the wrong shape raises with strict on and off, and nothing is loaded")
def wrong_shape(ctx):
    # Rejects no_shape_check (and Module.load_weights(strict=False), which lets a
    # wrong shape through) and updates_before_validating.
    for strict in (True, False):
        lm = MLXTinyLM(6, n_kv_heads=2)
        before = snapshot(lm)
        values = source_values(ctx, lm)
        tensors = as_file(values)
        # A (16, 32) weight stored as (32, 16).
        wrong = tensors.pop("layers.1.k_proj.weight")
        tensors["layers.1.k_proj.weight"] = wrong.T
        path = write(ctx, tensors)
        must_raise(ctx, lm, path, strict=strict)
        unchanged(lm, before)
        # A second file with the wrong shape elsewhere: an embedding one column short.
        tensors["layers.1.k_proj.weight"] = wrong
        tensors["embed.weight"] = mx.zeros((64, 31))
        path = write(ctx, tensors)
        must_raise(ctx, lm, path, strict=strict)
        unchanged(lm, before)


@case("the prefix is removed once, from the front; a tensor without it is unexpected")
def prefix_rules(ctx):
    # Rejects lstrip_prefix: "model.layers.0..." and "model.embed..." lose more than
    # the prefix when it is treated as a set of characters.
    lm = MLXTinyLM(7)
    before = snapshot(lm)
    values = source_values(ctx, lm)
    tensors = as_file(values, prefix="model.")
    path = write(ctx, tensors)
    check.equal(load(ctx, lm, path, prefix="model."), ([], []), label="result")
    check_parameters(lm, values, before)

    # The head is stored without the prefix: with prefix "model." it is not a tensor of
    # this module, so the module's own head is missing and the file's is unexpected.
    lm = MLXTinyLM(7)
    before = snapshot(lm)
    tensors["head.weight"] = tensors.pop("model.head.weight")
    path = write(ctx, tensors)
    must_raise(ctx, lm, path, prefix="model.")
    unchanged(lm, before)
    result = load(ctx, lm, path, prefix="model.", strict=False)
    check.equal(result, (["head.weight"], ["head.weight"]), label="(missing, unexpected)")
    check_parameters(lm, dict(values, **{"head.weight": before["head.weight"][0]}), before)

    # With no prefix at all, every prefixed name is unexpected and everything is missing.
    path = write(ctx, as_file(values, prefix="model."))
    missing, unexpected = load(ctx, MLXTinyLM(7), path, strict=False)
    check.equal(missing, sorted(values), label="missing")
    check.equal(unexpected, sorted("model." + name for name in values), label="unexpected")


@case("strict: one unexpected tensor alone, and one missing parameter alone, both raise")
def strict_each_way(ctx):
    # Rejects unexpected_ignored and silent_partial_load, one at a time.
    values = source_values(ctx, MLXTinyLM(8))

    lm = MLXTinyLM(8)
    before = snapshot(lm)
    extra = as_file(values)
    extra["layers.2.q_proj.weight"] = mx.zeros((32, 32))
    path = write(ctx, extra)
    must_raise(ctx, lm, path)
    unchanged(lm, before)
    result = load(ctx, lm, path, strict=False)
    check.equal(result, ([], ["layers.2.q_proj.weight"]), label="(missing, unexpected)")
    check_parameters(lm, values, before)

    lm = MLXTinyLM(8)
    before = snapshot(lm)
    partial = as_file(values)
    del partial["layers.0.norm2.weight"], partial["embed.weight"]
    path = write(ctx, partial)
    must_raise(ctx, lm, path)
    unchanged(lm, before)
    result = load(ctx, lm, path, strict=False)
    check.equal(result, (["embed.weight", "layers.0.norm2.weight"], []), label="result")
    kept = {name: before[name][0] for name in ("embed.weight", "layers.0.norm2.weight")}
    check_parameters(lm, dict(values, **kept), before)


@case("square matrices: a (32, 32) embedding is not a Linear weight, a (32, 32) Linear is")
def square_matrices(ctx):
    # Rejects transposes_every_matrix (the embedding has the right shape either way)
    # and never_transposes (so do the square projections).
    options = {"vocab_size": 32}
    lm = MLXTinyLM(9, **options)
    before = snapshot(lm)
    values = source_values(ctx, lm)
    path = write(ctx, as_file(values, transposed=linear_names(lm)))
    load(ctx, lm, path, transpose_linear=True)
    check_parameters(lm, values, before)
    check_logits(lm, values, **options)


@case("Linear layers in a list and a dict, biases, and a bare array parameter")
def nested_containers(ctx):
    net = Nested()
    before = snapshot(net)
    values = source_values(ctx, net)
    path = write(ctx, as_file(values, prefix="net.", transposed=linear_names(net)))
    result = load(ctx, net, path, prefix="net.", transpose_linear=True)
    check.equal(result, ([], []), label="(missing, unexpected)")
    check_parameters(net, values, before)

    oracle = Nested()
    oracle.update(tree_unflatten([(name, mx.array(value)) for name, value in values.items()]))
    x = mx.array(ctx.randn(3, 4))
    for index, (got, want) in enumerate(zip(net(x), oracle(x), strict=True)):
        check.allclose(np.array(got), np.array(want), atol=1e-5, label=f"output {index}")


@case(
    "a 32 MiB tensor the module does not use is never read from the file",
    requires="mlx-gpu",
)
def unused_tensor_not_read(ctx):
    # Rejects evaluates_everything: evaluating the whole file costs 32 MiB here.
    lm = MLXTinyLM(10)
    before = snapshot(lm)
    values = source_values(ctx, lm)
    tensors = as_file(values)
    tensors["vision_tower.patch_embed.weight"] = mx.zeros((4096, 2048))  # 32 MiB of float32
    path = write(ctx, tensors)
    del tensors
    with mlx_memory() as mem:
        result = ctx.solution.load_into(lm, path, strict=False)
        mx.eval(lm.parameters())  # a lazy load would otherwise measure nothing
    check.equal(result, ([], ["vision_tower.patch_embed.weight"]), label="(missing, unexpected)")
    check_parameters(lm, values, before)
    check.memory(mem, at_most_mib=8.0)
