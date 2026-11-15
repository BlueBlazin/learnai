"""Tests for cuda-device-placement.

On a CPU every tensor is on the CPU, so a tensor created in the wrong place, or a
value fetched to the host, changes nothing that a comparison of numbers can see. The
CPU cases therefore check `run` four ways:

1. the logits, against the definition computed here in float64 NumPy;
2. `DeviceSpy`: every `torch.arange`, `torch.zeros`, `torch.tensor`, ... called from
   the solution file must pass `device=`;
3. `SyncSpy`: the solution file makes no host read or transfer (`.cpu()`, `.item()`,
   `.tolist()`, `int(x)`, a boolean-mask index);
4. one case runs the whole forward pass on `torch.device("meta")`, whose tensors have
   a shape and no data: a CPU tensor that meets a meta tensor in an elementwise
   operation raises, as it would next to a CUDA tensor, whichever function created it,
   and so does every read of a value. That case is the one that sees a tensor used as
   a slice bound (`row[:n]`): torch reads the bound in C++ without calling
   `Tensor.__index__`, so `SyncSpy` does not count it.

`to_device` is tested by moving nested structures to the meta device, which costs
nothing and makes "this leaf was not moved" visible.

The GPU cases (`requires="cuda"`) were written from the PyTorch 2.15 documentation and
have never run on the machine that validates the catalog.
"""

import numpy as np
import torch

from whetstone.judge import case, check
from whetstone.kit import DeviceSpy, SyncSpy

CPU = torch.device("cpu")
META = torch.device("meta")


# --------------------------------------------------------------------------------------
# The forward pass of the statement
# --------------------------------------------------------------------------------------


def oracle(weights, tokens, lengths, bias):
    """Mean-pooled logits from the definition, in float64."""
    tok, pos, w_out = (weights[k].astype(np.float64) for k in ("tok_emb", "pos_emb", "w_out"))
    steps = tokens.shape[1]
    h = tok[tokens] + pos[np.arange(steps)]
    valid = np.arange(steps)[None, :] < lengths[:, None]
    pooled = (h * valid[:, :, None]).sum(axis=1) / lengths[:, None]
    return pooled @ w_out.T + bias.astype(np.float64)


def make(ctx, batch, steps, *, vocab=20, dim=8, classes=5, max_pos=32, lengths=None, dtype="f4"):
    """Random weights and a padded batch, as NumPy arrays."""
    weights = {
        "tok_emb": ctx.randn(vocab, dim).astype(dtype),
        "pos_emb": ctx.randn(max_pos, dim).astype(dtype),
        "w_out": ctx.randn(classes, dim).astype(dtype),
    }
    tokens = ctx.randint(0, vocab, (batch, steps))
    if lengths is None:
        lengths = ctx.randint(1, steps + 1, (batch,))
    lengths = np.asarray(lengths, dtype=np.int64)
    bias = ctx.randn(classes).astype(dtype)
    return weights, tokens, lengths, bias


def as_inputs(tokens, lengths, bias):
    return {
        "tokens": torch.from_numpy(tokens),
        "lengths": torch.from_numpy(lengths),
        "head": {"bias": torch.from_numpy(bias)},
    }


def check_logits(out, weights, tokens, lengths, bias):
    dtype = weights["w_out"].dtype
    check.that(isinstance(out, torch.Tensor), f"run: returned a {type(out).__name__}, not a tensor")
    check.that(
        tuple(out.shape) == (tokens.shape[0], weights["w_out"].shape[0]),
        f"run: got shape {tuple(out.shape)}, which is not the expected shape",
        f"expected (B, C) = {(tokens.shape[0], weights['w_out'].shape[0])}",
    )
    got_dtype = str(out.dtype).removeprefix("torch.")
    check.that(
        got_dtype == dtype.name,
        f"run: got dtype {got_dtype}, which is not the expected dtype",
        f"the result must have the dtype of the weights ({dtype.name})",
    )
    half = dtype == np.float16
    check.allclose(
        out.detach().cpu().float().numpy(),
        oracle(weights, tokens, lengths, bias).astype(np.float32),
        rtol=1e-2 if half else 1e-4,
        atol=3e-2 if half else 1e-5,
        label="run",
    )


def run_cpu(ctx, weights, tokens, lengths, bias):
    """Call the solution on the CPU: numbers first, then where tensors were made."""
    tensors = {name: torch.from_numpy(value) for name, value in weights.items()}
    with DeviceSpy() as factories, SyncSpy() as syncs:
        out = ctx.solution.run(tensors, as_inputs(tokens, lengths, bias), CPU)
    check_logits(out, weights, tokens, lengths, bias)
    check.that(
        not factories.implicit,
        f"run: {len(factories.implicit)} tensors were created without device= "
        "(they land on the default device, the CPU)",
        factories.describe(),
    )
    check.that(
        syncs.syncs == 0,
        f"run: {syncs.syncs} host synchronizations during the forward pass "
        "(a value was read or a tensor copied to the host)",
        syncs.describe(),
    )


def solution_line(ctx, exc):
    """' (line N of your file)' for the innermost frame of the solution on the traceback."""
    line, tb = None, exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == ctx.solution.__file__:
            line = tb.tb_lineno
        tb = tb.tb_next
    return f" (line {line} of your file)" if line is not None else ""


# --------------------------------------------------------------------------------------
# Nested structures for to_device
# --------------------------------------------------------------------------------------


def walk(tree, path="batch"):
    """Yield (path, leaf) for every leaf of a nested dict / list / tuple."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from walk(value, f"{path}[{key!r}]")
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            yield from walk(value, f"{path}[{index}]")
    else:
        yield path, tree


def shape_of(tree):
    """The structure without the leaves: container types, keys in order, lengths."""
    if isinstance(tree, dict):
        return ("dict", [(key, shape_of(value)) for key, value in tree.items()])
    if isinstance(tree, (list, tuple)):
        return (type(tree).__name__, [shape_of(value) for value in tree])
    return "tensor" if isinstance(tree, torch.Tensor) else type(tree).__name__


def check_moved(ctx, batch, device=META):
    """to_device(batch, device): structure kept, tensors moved, input untouched."""
    structure = shape_of(batch)
    leaves = list(walk(batch))
    out = ctx.solution.to_device(batch, device)

    check.that(
        shape_of(out) == structure,
        "to_device: the result does not have the structure of the input "
        "(container types, keys and their order, lengths)",
        f"input:  {structure}\nresult: {shape_of(out)}",
    )
    for (path, got), (_, original) in zip(walk(out), leaves, strict=True):
        if isinstance(original, torch.Tensor):
            check.that(
                got.device.type == device.type,
                f"to_device: the tensor at {path} is on {got.device.type}, not on the device",
            )
            check.that(
                got.dtype == original.dtype and got.shape == original.shape,
                f"to_device: the tensor at {path} changed its dtype or shape",
            )
        else:
            check.that(
                got is original,
                f"to_device: the leaf at {path}, which is not a tensor, was not passed through",
                f"expected the same object: {original!r}",
            )
    # The caller's batch is as it was: same leaves, on their original device.
    after = list(walk(batch))
    check.that(
        shape_of(batch) == structure
        and len(after) == len(leaves)
        and all(a is b for (_, a), (_, b) in zip(after, leaves, strict=True)),
        "to_device: the input structure was modified; it must return new containers",
    )
    return out


def random_tree(rng, depth=0):
    """A random nested batch: tensors, strings, numbers, and None at the leaves."""
    kind = rng.integers(0, 6) if depth < 3 else rng.integers(3, 6)
    if kind == 0:
        return {f"k{i}": random_tree(rng, depth + 1) for i in range(int(rng.integers(1, 4)))}
    if kind == 1:
        return [random_tree(rng, depth + 1) for _ in range(int(rng.integers(0, 4)))]
    if kind == 2:
        return tuple(random_tree(rng, depth + 1) for _ in range(int(rng.integers(1, 4))))
    if kind == 3:
        return [None, "request-17", 3, 0.5][int(rng.integers(0, 4))]
    dtype = [np.float32, np.int64, np.float16, np.bool_][int(rng.integers(0, 4))]
    shape = tuple(int(n) for n in rng.integers(1, 4, size=int(rng.integers(0, 3))))
    return torch.from_numpy(np.asarray(rng.integers(0, 2, size=shape)).astype(dtype))


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------


@case("the worked example: two sequences of lengths 3 and 1", public=True)
def worked_example(ctx):
    weights = {
        "tok_emb": np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32),
        "pos_emb": np.array([[0, 0], [0.5, 0.5], [1, 1]], dtype=np.float32),
        "w_out": np.array([[1, 0], [0, 1], [1, -1]], dtype=np.float32),
    }
    tokens = np.array([[1, 2, 3], [3, 0, 0]])
    lengths = np.array([3, 1])
    bias = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    want = [[1.2667, 1.3667, 0.3], [1.1, 1.2, 0.3]]
    if not np.allclose(oracle(weights, tokens, lengths, bias), want, atol=1e-4):
        raise RuntimeError("authoring error: the worked example changed")
    run_cpu(ctx, weights, tokens, lengths, bias)


@case("to_device on the batch of the example and on a deeper one", public=True)
def nested_batch(ctx):
    # Rejects shallow_to_device, tuples_become_lists, and mutates_input.
    batch = as_inputs(np.array([[1, 2, 3], [3, 0, 0]]), np.array([3, 1]), np.zeros(3, np.float32))
    check_moved(ctx, batch)
    deep = {
        "tokens": torch.zeros(2, 4, dtype=torch.int64),
        "request_ids": ["a17", "b03"],
        "images": [(torch.ones(3, 2), "left"), (torch.ones(3, 2), "right")],
        "head": {"bias": torch.zeros(5), "layers": [{"scale": torch.ones(8)}, None]},
        "temperature": 0.7,
    }
    out = check_moved(ctx, deep)
    check.that(
        isinstance(out["images"][0], tuple), "to_device: a tuple in the batch must stay a tuple"
    )
    # On the device where the tensors already are, the values are simply kept.
    same = check_moved(ctx, deep, CPU)
    check.equal(same["images"][1][0].numpy(), np.ones((3, 2), np.float32), label="a moved tensor")


@case("a random padded batch: logits, no tensor without device=, no host read", public=True)
def random_batch(ctx):
    # Rejects arange_default_device, cpu_round_trip, result_on_cpu, and counts_padding.
    run_cpu(ctx, *make(ctx, 4, 7))
    run_cpu(ctx, *make(ctx, 3, 5, lengths=[5, 1, 3]))


@case("float16 weights: the logits stay float16")
def half_precision(ctx):
    # Rejects hardcodes_float32: .float() promotes the activations to float32.
    run_cpu(ctx, *make(ctx, 4, 6, dtype="f2"))
    run_cpu(ctx, *make(ctx, 2, 9, classes=3, dtype="f2"))


@case("no padding at all, a single sequence, and sequences of length 1")
def edge_lengths(ctx):
    run_cpu(ctx, *make(ctx, 3, 6, lengths=[6, 6, 6]))
    run_cpu(ctx, *make(ctx, 1, 4, lengths=[2]))
    run_cpu(ctx, *make(ctx, 3, 5, lengths=[1, 1, 1]))
    run_cpu(ctx, *make(ctx, 2, 1, lengths=[1, 1]))


@case("to_device: a bare tensor, empty containers, and leaves that are not tensors")
def to_device_edges(ctx):
    out = ctx.solution.to_device(torch.ones(2, 3), META)
    check.that(
        isinstance(out, torch.Tensor) and out.device.type == "meta",
        "to_device: a bare tensor must come back as a tensor on the device",
    )
    check_moved(ctx, {"empty": [], "pair": (), "nothing": None, "nested": {"more": {}}})
    check_moved(ctx, [torch.ones(1), [torch.ones(2), [torch.ones(3), ("x", torch.ones(4))]]])
    check_moved(ctx, (torch.zeros(2, dtype=torch.bool), {"ids": torch.arange(3)}))


@case("to_device on random nested structures")
def to_device_random(ctx):
    def prop(rng):
        check_moved(ctx, {"root": random_tree(rng), "tensor": torch.ones(2)})

    check.forall(prop, trials=20)


@case("the forward pass on a data-free device: every operand must be on the device")
def on_meta_device(ctx):
    # Rejects positions_from_numpy (and every other tensor that stays on the CPU,
    # however it was created): CPU and meta operands do not mix. Rejects
    # loops_over_rows: a slice bound that is a tensor is a host read.
    weights, tokens, lengths, bias = make(ctx, 3, 6)
    tensors = {name: torch.from_numpy(value).to(META) for name, value in weights.items()}
    try:
        out = ctx.solution.run(tensors, as_inputs(tokens, lengths, bias), META)
    except (RuntimeError, NotImplementedError) as exc:
        text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        if "device" not in text and "meta tensor" not in text:
            raise
        line = solution_line(ctx, exc)
        check.that(
            False,
            f"run: not every tensor stayed on `device`{line}: {text}",
            "The weights were on torch.device('meta') and the batch on the CPU. A tensor "
            "that run created or left on the CPU met a tensor on the device, or a tensor "
            "was copied back to the host (a meta tensor has no data to copy).",
        )
    check.that(isinstance(out, torch.Tensor), "run: must return a tensor")
    check.that(out.device.type == "meta", f"run: the result is on {out.device.type}, not on device")
    check.that(
        tuple(out.shape) == (3, 5) and out.dtype == torch.float32,
        f"run: got shape {tuple(out.shape)} and dtype {out.dtype}, which is not what is expected",
    )


@case("many batches of different sizes")
def sweep(ctx):
    for batch, steps, vocab, dim, classes in [
        (1, 1, 4, 2, 2),
        (5, 12, 30, 16, 7),
        (8, 3, 6, 4, 1),
        (2, 20, 50, 8, 4),
    ]:
        run_cpu(ctx, *make(ctx, batch, steps, vocab=vocab, dim=dim, classes=classes))


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: a CPU batch and CUDA weights give logits on the CUDA device", requires="cuda")
def gpu_run(ctx):
    device = torch.device("cuda")
    for dtype in ("f4", "f2"):
        weights, tokens, lengths, bias = make(ctx, 4, 7, dtype=dtype)
        tensors = {name: torch.from_numpy(value).to(device) for name, value in weights.items()}
        with DeviceSpy() as factories, SyncSpy() as syncs:
            out = ctx.solution.run(tensors, as_inputs(tokens, lengths, bias), device)
        check.that(isinstance(out, torch.Tensor), "run: must return a tensor")
        check.that(out.is_cuda, f"run: the result is on {out.device}, not on the CUDA device")
        check_logits(out, weights, tokens, lengths, bias)
        check.that(
            not factories.implicit,
            f"run: {len(factories.implicit)} tensors were created without device=",
            factories.describe(),
        )
        check.that(
            syncs.syncs == 0,
            f"run: {syncs.syncs} host synchronizations during the forward pass",
            syncs.describe(),
        )


@case("on the GPU: to_device moves every leaf of a nested batch", requires="cuda")
def gpu_to_device(ctx):
    device = torch.device("cuda")
    deep = {
        "tokens": torch.arange(8).reshape(2, 4),
        "request_ids": ["a17", "b03"],
        "images": [(torch.ones(3, 2), "left"), (torch.full((3, 2), 2.0), "right")],
        "head": {"bias": torch.zeros(5), "layers": [{"scale": torch.ones(8)}, None]},
    }
    out = check_moved(ctx, deep, device)
    check.equal(out["tokens"].cpu().numpy(), np.arange(8).reshape(2, 4), label="a moved tensor")
    check.allclose(out["images"][1][0].cpu().numpy(), np.full((3, 2), 2.0), label="a moved tensor")
