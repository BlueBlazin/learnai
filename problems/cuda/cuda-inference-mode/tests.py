"""Tests for cuda-inference-mode.

The CPU cases decide validation. Each call of `predict` is checked four ways:

1. the numbers, against the same model run here in evaluation mode (a deep copy, so
   the model handed to the solution is never touched by the test);
2. that the result was created under inference mode (`Tensor.is_inference()`), which
   tells `torch.inference_mode()` from `torch.no_grad()` and from `.detach()`;
3. that every submodule has the `training` flag it had before, and that no buffer
   (BatchNorm's running statistics) changed;
4. that the caller's autograd mode is what it was.

`torch.device("meta")` is the second device every machine has: its tensors have a
shape and no data. One case uses it as the target to see whether `x` is moved to the
device (the model in that case only records where its input lives; the first real
operation on a meta tensor costs torch about two seconds of imports).

The two GPU cases (`requires="cuda"`) were written from the PyTorch 2.15
documentation and have never run on the machine that validates the catalog.
"""

import copy
import inspect

import numpy as np
import torch
from torch import nn

from learnai.judge import case, check

CPU = torch.device("cpu")


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def seeded(model, seed):
    """Fill parameters and running statistics with seeded values; returns the model."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.5)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                # Far from a batch's own statistics, so training mode is visible.
                module.running_mean.copy_(torch.randn(module.num_features, generator=gen))
                module.running_var.copy_(torch.rand(module.num_features, generator=gen) + 0.5)
    return model


def mlp(seed, sizes=(6, 16, 4), p=0.5, norm=False):
    """Linear, optional BatchNorm, ReLU, Dropout per hidden layer; then a Linear head."""
    layers = []
    for index in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if norm:
            layers.append(nn.BatchNorm1d(sizes[index + 1]))
        layers += [nn.ReLU(), nn.Dropout(p)]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return seeded(nn.Sequential(*layers), seed)


class Failing(nn.Module):
    """A layer that rejects its input, as a shape check or a tokenizer bug would."""

    def forward(self, x):
        raise ValueError("the model rejected this batch")


class DeviceProbe(nn.Module):
    """Records where its input lives and returns it; runs no tensor operation."""

    seen = None

    def forward(self, x):
        self.seen = x.device
        return x


def eval_output(model, x):
    """What the statement defines: the model in evaluation mode, on a private copy."""
    twin = copy.deepcopy(model).eval()
    with torch.no_grad():
        return twin(torch.from_numpy(x)).numpy()


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def flags_of(model):
    return [module.training for module in model.modules()]


def check_caller_mode(label):
    """The caller's autograd mode must be untouched: gradients on, inference mode off."""
    grad, inference = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
    torch.set_grad_enabled(True)  # do not let one failure poison the cases that follow
    check.that(
        grad and not inference,
        f"{label}: the caller's autograd mode was changed "
        f"(gradients enabled: {grad}, inference mode: {inference})",
        "After the call the caller must be where it was: gradients enabled and inference "
        "mode off. A context manager restores the mode; torch.set_grad_enabled(False) "
        "as a statement does not.",
    )


def check_flags(model, before, label="predict"):
    after = flags_of(model)
    changed = sum(a != b for a, b in zip(after, before, strict=True))
    check.that(
        changed == 0,
        f"{label}: the training flag of {changed} of {len(before)} modules "
        "(the model and its submodules) is not what it was before the call",
        f"before: {before}\nafter:  {after}\n(in the order of model.modules())",
    )


def check_inference_tensor(out, label="predict"):
    check.that(
        isinstance(out, torch.Tensor), f"{label}: returned a {type(out).__name__}, not a tensor"
    )
    check.that(
        not out.requires_grad and out.grad_fn is None,
        f"{label}: the result is attached to an autograd graph",
        f"requires_grad={out.requires_grad}, grad_fn={out.grad_fn}",
    )
    check.that(
        out.is_inference(),
        f"{label}: the result was not created under inference mode",
        "Tensor.is_inference() is False. torch.no_grad() and .detach() return ordinary "
        "tensors; only torch.inference_mode() makes inference tensors.",
    )


def run_predict(ctx, model, x):
    """Call the solution on the CPU and check result, mode, flags, and buffers."""
    ctx.defer(lambda: torch.set_grad_enabled(True))
    want = eval_output(model, x)
    before = flags_of(model)
    buffers = {name: buf.clone() for name, buf in model.named_buffers()}
    out = ctx.solution.predict(model, torch.from_numpy(x), CPU)
    check_caller_mode("predict")
    check_inference_tensor(out)
    check.allclose(out.numpy(), want, atol=1e-5, label="predict", dtype=True)
    check_flags(model, before)
    stale = [name for name, buf in model.named_buffers() if not torch.equal(buf, buffers[name])]
    check.that(
        not stale,
        f"predict: {len(stale)} buffers of the model changed during the call",
        f"changed: {stale}\nBatchNorm updates its running statistics whenever it runs in "
        "training mode, also under inference mode.",
    )
    return out


# --------------------------------------------------------------------------------------
# CPU cases: predict
# --------------------------------------------------------------------------------------


@case("the worked example: a linear layer and dropout, handed over in training mode", public=True)
def worked_example(ctx):
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
        model[0].bias.copy_(torch.tensor([0.5, -0.5]))
    model.train()
    x = np.array([[2.0, 4.0]], dtype=np.float32)
    out = run_predict(ctx, model, x)
    if not np.allclose(out.numpy(), [[2.5, 5.5]]):
        raise RuntimeError("authoring error: the worked example changed")
    check.that(model.training, "predict: the model was in training mode and must still be")


@case("a model in training mode and the same model in evaluation mode", public=True)
def both_modes(ctx):
    x = ctx.randn(5, 6)
    # Rejects stays_in_training_mode (dropout active) and eval_not_restored.
    run_predict(ctx, mlp(1).train(), x)
    run_predict(ctx, mlp(1).eval(), x)


@case("the result is an inference tensor although parameters and input require grad", public=True)
def no_graph(ctx):
    # Rejects no_grad_only, detach_after_forward, and global_grad_switch.
    model = mlp(2).train()
    x = torch.from_numpy(ctx.randn(3, 6)).requires_grad_(True)
    ctx.defer(lambda: torch.set_grad_enabled(True))
    before = flags_of(model)
    out = ctx.solution.predict(model, x, CPU)
    check_caller_mode("predict")
    check_inference_tensor(out)
    check.allclose(out.numpy(), eval_output(model, x.detach().numpy()), atol=1e-5, label="predict")
    check_flags(model, before)
    # The caller can still train afterward.
    loss = model(x).sum()
    check.that(loss.requires_grad, "predict: autograd no longer records after the call")


@case("the model raises: the error propagates and the model is left as it was", public=True)
def model_raises(ctx):
    # Rejects restore_skipped_on_error.
    ctx.defer(lambda: torch.set_grad_enabled(True))
    model = nn.Sequential(nn.Linear(6, 6), nn.Dropout(0.5), Failing()).train()
    before = flags_of(model)
    x = torch.from_numpy(ctx.randn(2, 6))
    check.raises(ValueError, ctx.solution.predict, model, x, CPU, label="predict")
    check_caller_mode("predict")
    check_flags(model, before)


@case("batch normalization uses its running statistics and does not update them")
def batchnorm_statistics(ctx):
    x = ctx.randn(8, 6, loc=3.0)
    run_predict(ctx, mlp(3, norm=True).train(), x)
    run_predict(ctx, mlp(4, sizes=(6, 12, 12, 3), norm=True).train(), x)


@case("a submodule frozen in evaluation mode inside a model that is training")
def mixed_flags(ctx):
    # Rejects restores_root_flag: model.train(True) would thaw the frozen layers.
    model = mlp(5, sizes=(6, 10, 10, 4), norm=True).train()
    model[1].eval()  # a frozen BatchNorm, as in fine-tuning
    model[3].eval()  # and a disabled dropout
    run_predict(ctx, model, ctx.randn(4, 6))
    check.that(
        model.training and not model[1].training and not model[3].training,
        "predict: a submodule's training flag differs from before the call",
    )
    # The other way around: the root in evaluation mode, one layer left training.
    model = mlp(6, sizes=(6, 10, 4)).eval()
    model[2].train()
    run_predict(ctx, model, ctx.randn(4, 6))


@case("x is moved to the device before the model sees it (a data-free device stands in)")
def input_is_moved(ctx):
    # Rejects input_not_moved. On the CPU every tensor is on the CPU, so the target
    # here is the meta device; the probe runs no operation on it.
    ctx.defer(lambda: torch.set_grad_enabled(True))
    meta = torch.device("meta")
    model = nn.Sequential(DeviceProbe()).train()
    before = flags_of(model)
    out = ctx.solution.predict(model, torch.from_numpy(ctx.randn(3, 6)), meta)
    check_caller_mode("predict")
    seen = model[0].seen
    check.that(
        seen is not None and seen.type == "meta",
        "predict: the model received an input that is not on `device`",
        f"device was 'meta'; the model's forward saw x on {seen}",
    )
    check.that(isinstance(out, torch.Tensor), "predict: must return a tensor")
    check.that(
        out.device.type == "meta", f"predict: the result is on {out.device.type}, not on `device`"
    )
    check.that(
        tuple(out.shape) == (3, 6),
        f"predict: got shape {tuple(out.shape)}, which is not the expected shape",
    )
    check_flags(model, before)


@case("random models with random training flags: numbers, flags, and mode")
def random_models(ctx):
    def prop(rng):
        hidden = [int(n) for n in rng.integers(4, 12, size=int(rng.integers(1, 4)))]
        model = mlp(int(rng.integers(0, 10_000)), sizes=(6, *hidden, 3), norm=bool(rng.integers(2)))
        for module in model.modules():
            module.training = bool(rng.integers(2))
        run_predict(ctx, model, ctx.randn(int(rng.integers(2, 6)), 6))

    check.forall(prop, trials=8)


# --------------------------------------------------------------------------------------
# CPU cases: as_inference
# --------------------------------------------------------------------------------------


@case("as_inference on a plain function: arguments pass through, results are inference tensors")
def wrapped_function(ctx):
    ctx.defer(lambda: torch.set_grad_enabled(True))
    weight = torch.tensor([1.0, -2.0, 0.5], requires_grad=True)

    def scale(x, factor=1.0, *, shift=0.0):
        return weight * x * factor + shift, "tag"

    wrapped = ctx.solution.as_inference(scale)
    check.that(callable(wrapped), "as_inference: must return a callable")
    x = torch.from_numpy(ctx.randn(3))
    result = wrapped(x, 3.0, shift=1.0)
    check_caller_mode("as_inference")
    check.that(
        isinstance(result, tuple) and len(result) == 2 and result[1] == "tag",
        "as_inference: the wrapped function's return value was not passed through",
    )
    check_inference_tensor(result[0], "as_inference")
    want = weight.detach().numpy() * x.numpy() * 3.0 + 1.0
    check.allclose(result[0].numpy(), want, label="as_inference")

    def fails(x):
        raise KeyError("missing")

    check.raises(KeyError, ctx.solution.as_inference(fails), x, label="as_inference")
    check_caller_mode("as_inference")


@case("as_inference on a generator function: every step runs under inference mode")
def wrapped_generator(ctx):
    # Rejects wrapper_returns_generator (steps run with autograd on) and
    # generator_leaks_mode (the caller is in inference mode between steps).
    ctx.defer(lambda: torch.set_grad_enabled(True))
    layer = seeded(nn.Linear(4, 4), 11)

    def stream(x, steps):
        for _ in range(steps):
            x = torch.tanh(layer(x))
            yield x

    x = torch.from_numpy(ctx.randn(2, 4))
    with torch.no_grad():
        want, state = [], x
        for _ in range(3):
            state = torch.tanh(layer(state))
            want.append(state.numpy())

    iterator = ctx.solution.as_inference(stream)(x, 3)
    check.that(
        inspect.isgenerator(iterator) or hasattr(iterator, "__next__"),
        "as_inference: calling the wrapped generator function must return an iterator",
    )
    got = []
    try:
        for item in iterator:
            leaked = torch.is_inference_mode_enabled() or not torch.is_grad_enabled()
            check.that(
                not leaked,
                "as_inference: between two steps of the generator the caller's own code "
                "runs with autograd disabled",
                "The mode must be entered when the generator resumes and left when it "
                "yields; a with block around the whole generator stays open while the "
                "generator is suspended.",
            )
            check_inference_tensor(item, "as_inference (a yielded tensor)")
            got.append(item.numpy())
    finally:
        if hasattr(iterator, "close"):
            iterator.close()
    check_caller_mode("as_inference")
    check.equal(len(got), 3, label="number of yielded tensors")
    for index, (g, w) in enumerate(zip(got, want, strict=True)):
        check.allclose(g, w, atol=1e-5, label=f"yielded tensor {index}")


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: the result is an inference tensor on the device", requires="cuda")
def gpu_result(ctx):
    ctx.defer(lambda: torch.set_grad_enabled(True))
    device = torch.device("cuda")
    x = ctx.randn(5, 6)
    cpu_model = mlp(1, norm=True).train()
    want = eval_output(cpu_model, x)
    model = copy.deepcopy(cpu_model).to(device)
    before = flags_of(model)
    out = ctx.solution.predict(model, torch.from_numpy(x), device)
    check_caller_mode("predict")
    check_inference_tensor(out)
    check.that(out.is_cuda, f"predict: the result is on {out.device}, not on the CUDA device")
    check.allclose(out.cpu().numpy(), want, rtol=1e-3, atol=1e-4, label="predict")
    check_flags(model, before)


@case("on the GPU: peak memory is well below a pass that records the graph", requires="cuda")
def gpu_peak_memory(ctx):
    # 12 hidden layers of 4096 x 512 float32 activations (8 MiB each). A pass that
    # records the graph keeps every ReLU output alive for backward (about 100 MiB);
    # inference mode frees each activation as soon as the next layer has consumed it
    # (two or three alive at a time).
    ctx.defer(lambda: torch.set_grad_enabled(True))
    device = torch.device("cuda")
    layers = []
    for _ in range(12):
        layers += [nn.Linear(512, 512), nn.ReLU()]
    model = seeded(nn.Sequential(*layers), 0).to(device).train()
    x = torch.from_numpy(ctx.randn(4096, 512))

    def peak_bytes(run):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        out = run()
        torch.cuda.synchronize(device)
        used = torch.cuda.max_memory_allocated(device) - before
        del out
        return used

    with torch.no_grad():  # warm-up: library handles and allocator pools
        model(x.to(device))
    recorded = peak_bytes(lambda: model(x.to(device)))
    used = peak_bytes(lambda: ctx.solution.predict(model, x, device))
    check_caller_mode("predict")
    ctx.metric("peak_memory", used / 2**20, "MiB")
    check.that(
        used <= 0.6 * recorded,
        f"predict: peak GPU memory was {used / 2**20:.0f} MiB, more than 60% of what a "
        "forward pass that records the autograd graph needs",
        f"with the graph recorded: {recorded / 2**20:.0f} MiB (torch.cuda.max_memory_allocated)",
    )
