"""Tests for mlx-quantize-module-tree.

`quantize_model` is judged against MLX's own mechanism: every case builds the same
float model twice from one seed, lets the solution quantize one copy, and quantizes
the twin here with `nn.quantize(twin, class_predicate=...)`. The two trees must then
agree layer for layer (types, bit widths, group sizes, parameters) and produce the
same output. Further checks on every call:

- the recipe was called exactly once for every `nn.Linear` and `nn.Embedding`, with
  its dotted path and the module itself, and for nothing else;
- the returned dict lists exactly the layers that were replaced;
- layers the recipe declined are the very objects they were;
- every parameter is evaluated on return (`mx.export_to_dot` lists no pending
  operation), so no lazy `Quantize` node keeps a float weight alive.

`bits_per_weight` is compared with a count made here: the bytes of the parameters,
times 8, over the number of weights of the *float* twin (quantizing changes how
weights are stored, not how many there are).

One case needs the Apple GPU: a model with 24 MiB of float weights must hold under
8 MiB after `quantize_model` returns, with the test evaluating nothing.

The recipe is handed over behind the kit's `Guarded`: a solution may call it. The
function it wraps and its record of the calls are refused. The model is not guarded:
nothing is counted on it, and the solution replaces its layers.
"""

import io

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from learnai.judge import case, check
from learnai.kit import Guarded, MLXTinyLM, check_guard, guard, mlx_memory
from mlx import nn

FLOAT_LAYERS = (nn.Linear, nn.Embedding)
QUANTIZED_LAYERS = (nn.QuantizedLinear, nn.QuantizedEmbedding)
TOKENS = [0, 17, 30, 5, 9, 23]


class Block(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.attn = nn.Linear(dims, dims, bias=False)
        self.mlp = nn.Linear(dims, dims)
        self.norm = nn.RMSNorm(dims)

    def __call__(self, x):
        x = x + self.attn(self.norm(x))
        return x + nn.silu(self.mlp(x))


class Tiny(nn.Module):
    """The model of the worked example: an embedding, two blocks in a list, a head."""

    def __init__(self, vocab=32, dims=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, dims)
        self.blocks = [Block(dims), Block(dims)]
        self.head = nn.Linear(dims, vocab, bias=False)

    def __call__(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class Deep(nn.Module):
    """Layers in lists of lists and dicts of dicts, and a parameter on a container."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(64, 64)
        self.stages = [[nn.Linear(64, 64, bias=False), nn.Linear(64, 128)], [nn.Linear(128, 64)]]
        self.heads = {
            "a": nn.Linear(64, 32),
            "b": {"proj": nn.Linear(64, 64, bias=False), "table": nn.Embedding(16, 64)},
        }
        self.gain = mx.ones((64,))
        self.inner = Block(64)

    def __call__(self, ids):
        x = self.stem(self.heads["b"]["table"](ids)) * self.gain
        x = self.stages[0][1](nn.silu(self.stages[0][0](x)))
        x = self.inner(self.stages[1][0](x))
        return self.heads["a"](x), self.heads["b"]["proj"](x)


def seeded(model, seed):
    """Give every parameter of `model` seeded float32 values, and evaluate them."""
    rng = np.random.default_rng([seed, 0x51])

    def fill(p):
        return mx.array(rng.standard_normal(p.shape).astype(np.float32) * 0.2)

    model.update(tree_map(fill, model.parameters()))
    mx.eval(model.parameters())
    return model


def pending_ops(tree):
    """Number of operations not yet computed behind the arrays in `tree` (0: evaluated)."""
    arrays = [leaf for _, leaf in tree_flatten([tree]) if isinstance(leaf, mx.array)]
    if not arrays:
        return 0
    dot = io.StringIO()
    mx.export_to_dot(dot, *arrays)
    return dot.getvalue().count("shape=rectangle")


def float_layers(model):
    return {p: m for p, m in model.named_modules() if isinstance(m, FLOAT_LAYERS)}


def n_parameters(model):
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def n_bytes(model):
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


class Recipe(Guarded):
    """Wraps a recipe function and records every call."""

    label = "recipe"
    refused = Guarded.refused | {"fn", "calls"}

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, path, module):
        self.calls.append((path, module))
        params = self.fn(path, module)
        return None if params is None else dict(params)


def by_path(table, default=None):
    """A recipe that looks the path up in a dict."""
    return lambda path, module: table.get(path, default)


def reference_quantize(twin, fn):
    """The same recipe through MLX's own nn.quantize and its class_predicate."""

    def predicate(path, module):
        if not isinstance(module, FLOAT_LAYERS):
            return False
        params = fn(path, module)
        return False if params is None else dict(params)

    nn.quantize(twin, class_predicate=predicate)
    mx.eval(twin.parameters())
    return twin


def run(ctx, make, fn, forward):
    """Quantize a fresh model with the solution and its twin with nn.quantize; compare."""
    model, twin = make(), make()
    before = float_layers(model)
    want = {p: dict(params) for p, m in before.items() if (params := fn(p, m)) is not None}
    recipe = Recipe(fn)
    guard(ctx)
    result = ctx.solution.quantize_model(model, recipe)
    check_guard()
    # Measured first: everything below may evaluate the parameters.
    left_lazy = pending_ops(model.parameters())

    # The recipe saw every Linear and Embedding once, and nothing else.
    called = sorted(path for path, _ in recipe.calls)
    check.that(
        called == sorted(before),
        f"quantize_model: the recipe was called for {len(called)} paths, expected "
        f"{len(before)}: the nn.Linear and nn.Embedding layers of the model, once each",
        f"called for {called}\nexpected {sorted(before)}",
    )
    check.that(
        all(module is before[path] for path, module in recipe.calls),
        "quantize_model: the recipe was not given the module that sits at the path",
    )

    # The tree: replaced where the recipe said so, untouched elsewhere.
    after = dict(model.named_modules())
    for path, module in before.items():
        now = after.get(path)
        if path in want:
            kind = nn.QuantizedLinear if isinstance(module, nn.Linear) else nn.QuantizedEmbedding
            check.that(
                type(now) is kind,
                f"quantize_model: the layer at {path} is a {type(now).__name__} after the "
                f"call, expected a {kind.__name__}, as the recipe asked",
                f"expected {kind.__name__} with {want[path]}",
            )
            got = {"bits": now.bits, "group_size": now.group_size}
            check.that(
                got == want[path] and now.mode == "affine",
                f"quantize_model: the layer at {path} was quantized with {got}, mode "
                f"{now.mode!r}, expected {want[path]}, mode 'affine'",
                f"expected {want[path]}, affine",
            )
        else:
            check.that(
                now is module,
                f"quantize_model: the layer at {path} was replaced or changed although the "
                "recipe returned None for it",
            )
    check.that(isinstance(result, dict), "quantize_model: must return a dict")
    check.equal(result, want, label="returned dict")

    # Parameters and outputs, against nn.quantize on the twin.
    reference_quantize(twin, fn)
    got_params = dict(tree_flatten(model.parameters()))
    want_params = dict(tree_flatten(twin.parameters()))
    check.that(
        sorted(got_params) == sorted(want_params),
        "quantize_model: the parameter names of the model are not the expected names",
        f"got {sorted(got_params)}\nexpected {sorted(want_params)}",
    )
    for name, expected in want_params.items():
        value = got_params[name]
        if mx.issubdtype(expected.dtype, mx.floating):
            check.allclose(np.array(value), np.array(expected), label=f"parameter {name}")
        else:
            check.equal(np.array(value), np.array(expected), label=f"parameter {name}")
    check.allclose(
        np.array(forward(model)), np.array(forward(twin)), rtol=1e-4, atol=1e-4, label="output"
    )

    check.that(
        left_lazy == 0,
        f"quantize_model: returned with unevaluated parameters ({left_lazy} operations "
        "still pending)",
        "a lazy mx.quantize keeps the float weight alive; evaluate the model's parameters",
    )
    return model, twin


def check_bpw(ctx, model, float_twin, label="bits_per_weight"):
    got = ctx.solution.bits_per_weight(model)
    check.that(type(got) is float, "bits_per_weight: must return a Python float")
    want = 8 * n_bytes(model) / n_parameters(float_twin)
    check.allclose(np.float64(got), np.float64(want), rtol=1e-6, label=label)
    return got


def lm_forward(model):
    return model(mx.array(TOKENS))[0]


def tiny_forward(model):
    return model(mx.array(TOKENS))


def deep_forward(model):
    a, b = model(mx.array([0, 3, 15, 7]))
    return mx.concatenate([a, b], axis=-1)


EXAMPLE_RECIPE = {
    "embed": {"bits": 8, "group_size": 64},
    "head": {"bits": 8, "group_size": 64},
    "blocks.0.attn": {"bits": 4, "group_size": 64},
    "blocks.0.mlp": {"bits": 4, "group_size": 64},
    "blocks.1.attn": {"bits": 4, "group_size": 64},
}


def mixed_lm_recipe(path, module):
    """8 bits for the embedding and the head, 4 bits elsewhere; groups of 32 or 64."""
    if path in ("embed", "head"):
        return {"bits": 8, "group_size": 64}
    return {"bits": 4, "group_size": 64 if module.weight.shape[-1] > 64 else 32}


@case("the worked example: a mixed 4/8-bit recipe that leaves one layer float", public=True)
def worked_example(ctx):
    def make():
        return seeded(Tiny(), 1)

    model, _ = run(ctx, make, by_path(EXAMPLE_RECIPE), tiny_forward)
    if n_bytes(model) != 29696:
        raise RuntimeError("authoring error: the worked example changed")
    got = check_bpw(ctx, model, make())
    check.allclose(np.float64(got), np.float64(29696 * 8 / 20736), rtol=1e-6)
    check_bpw(ctx, make(), make(), label="bits_per_weight of the float model")


@case("a transformer: layers inside a list, 8 bits for embedding and head", public=True)
def transformer_in_a_list(ctx):
    # Rejects skips_lists (the blocks sit in `layers`), embedding_ignored,
    # ignores_recipe_params, and leaf_name_paths.
    def make():
        return MLXTinyLM(3, d_model=64)

    model, _ = run(ctx, make, mixed_lm_recipe, lm_forward)
    check_bpw(ctx, model, make())
    check.at_most("parameter_bytes", n_bytes(model), 69888, unit="bytes")


@case("a recipe that declines everything changes nothing", public=True)
def declines_everything(ctx):
    # Rejects quantizes_excluded.
    def make():
        return seeded(Tiny(), 2)

    model, _ = run(ctx, make, lambda path, module: None, tiny_forward)
    check_bpw(ctx, model, make())


# Hidden cases.


@case("layers in lists of lists and dicts of dicts")
def nested_containers(ctx):
    def make():
        return seeded(Deep(), 4)

    def fn(path, module):
        if path.startswith("stages.") or path == "heads.b.table":
            return {"bits": 4, "group_size": 64}
        if path == "heads.b.proj":
            return {"bits": 8, "group_size": 32}
        return None

    model, _ = run(ctx, make, fn, deep_forward)
    # The container's own parameter `gain` is 64 weights like any others.
    check_bpw(ctx, model, make())
    run(ctx, make, lambda path, module: {"bits": 2, "group_size": 32}, deep_forward)


@case("only the embedding; only one layer deep in the tree")
def single_layers(ctx):
    def make():
        return MLXTinyLM(5, d_model=64)

    model, _ = run(ctx, make, by_path({"embed": {"bits": 4, "group_size": 32}}), lm_forward)
    check_bpw(ctx, model, make())
    one = {"layers.1.gate_proj": {"bits": 8, "group_size": 64}}
    model, _ = run(ctx, make, by_path(one), lm_forward)
    check_bpw(ctx, model, make())


@case("a recipe that decides by the module: layers with 128 inputs only")
def decides_by_module(ctx):
    def make():
        return MLXTinyLM(6, d_model=64, n_layers=3)

    def fn(path, module):
        if isinstance(module, nn.Linear) and module.weight.shape[1] == 128:
            return {"bits": 4, "group_size": 128}
        return None

    model, _ = run(ctx, make, fn, lm_forward)
    check_bpw(ctx, model, make())


@case("a model that is already partly quantized: the quantized layers are left alone")
def already_quantized(ctx):
    first = by_path({"blocks.0.attn": {"bits": 8, "group_size": 64}})

    def make():
        return reference_quantize(seeded(Tiny(), 7), first)

    everything = by_path({}, default={"bits": 4, "group_size": 32})
    model, _ = run(ctx, make, everything, tiny_forward)
    check.that(
        model.blocks[0].attn.bits == 8,
        "quantize_model: a layer that was already quantized was quantized again",
    )
    check_bpw(ctx, model, seeded(Tiny(), 7))


@case("bits_per_weight of float32, float16, and uniformly quantized models")
def bits_per_weight_only(ctx):
    # Independent of the solution's quantize_model: the models are quantized here.
    # Rejects bpw_ignores_scales and bpw_counts_stored_elements.
    def make():
        return MLXTinyLM(8, d_model=64)

    check.allclose(np.float64(check_bpw(ctx, make(), make())), 32.0, rtol=1e-6)
    half = MLXTinyLM(8, d_model=64, dtype=mx.float16)
    check.allclose(np.float64(check_bpw(ctx, half, make())), 16.0, rtol=1e-6)
    for bits, group_size in [(4, 64), (8, 64), (4, 32), (2, 32)]:
        uniform = by_path({}, default={"bits": bits, "group_size": group_size})
        check_bpw(ctx, reference_quantize(make(), uniform), make())
    deep = reference_quantize(seeded(Deep(), 9), by_path({"stem": {"bits": 4, "group_size": 64}}))
    check_bpw(ctx, deep, seeded(Deep(), 9))


@case("parameter bytes of the mixed recipe on a wider model: at most 0.2 of the float model")
def mixed_recipe_bytes(ctx):
    def make():
        return MLXTinyLM(10, vocab_size=256, d_model=128, n_layers=3)

    model, _ = run(ctx, make, mixed_lm_recipe, lm_forward)
    check.at_most("parameter_bytes_ratio", round(n_bytes(model) / n_bytes(make()), 4), 0.2)
    check_bpw(ctx, model, make())


@case(
    "24 MiB of float weights: after quantize_model returns, the model holds under 8 MiB",
    requires="mlx-gpu",
)
def float_weights_are_freed(ctx):
    # Rejects left_lazy by what it costs: lazy codes keep every float weight alive.
    def fn(path, module):
        return {"bits": 4, "group_size": 64}

    with mlx_memory() as mem:
        model = MLXTinyLM(11, vocab_size=1024, d_model=512, n_layers=2, n_heads=8)
        result = ctx.solution.quantize_model(model, fn)
    check.memory(mem, at_most_mib=8.0, what="active", metric="model_memory")
    check.that(len(result) == 16, "quantize_model: the returned dict does not list every layer")
    twin = reference_quantize(
        MLXTinyLM(11, vocab_size=1024, d_model=512, n_layers=2, n_heads=8), fn
    )
    check.allclose(
        np.array(lm_forward(model)),
        np.array(lm_forward(twin)),
        rtol=1e-3,
        atol=1e-3,
        label="output",
    )
