"""Tests for cuda-autocast-inference.

The model handed to the solution is a `Probe`: a small torch module that, each time it
is called, records whether an autocast region was active for its device, the 16-bit
type of that region, and the logits it returned (their dtype and their values). That
makes the three things a comparison of numbers cannot see observable on a CPU:

- the body ran under autocast at all (rejects a missing region, a region for the
  wrong `device_type`, and a model converted with `.to(dtype)`);
- the region used the requested dtype;
- the statistics were computed in float32 from exactly the logits the body produced:
  the expected log-probabilities and entropies are computed here in float64 from the
  recorded 16-bit logits, so any arithmetic done in the 16-bit type shows up as an
  error a thousand times the tolerance, and as NaN when a float16 quotient overflows.

Facts about the CPU this file relies on, verified on torch 2.15: `torch.autocast("cpu",
dtype=...)` accepts bfloat16 and float16; `nn.Linear` then returns the 16-bit type;
`log_softmax`, `/`, and `sum` of a 16-bit tensor stay 16-bit inside and outside the
region; `torch.autocast("cuda", ...)` without CUDA warns and disables itself.

The GPU cases (`requires="cuda"`) were written from the PyTorch 2.15 documentation and
have never run on the machine that validates the catalog.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from learnai.judge import case, check

FN = "forward_amp"
CPU = torch.device("cpu")
BF16, F16 = torch.bfloat16, torch.float16


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------


@dataclass
class Call:
    autocast: bool  # was an autocast region active for the model's device type?
    region_dtype: torch.dtype  # the 16-bit type of that region
    logits: torch.Tensor  # what the body returned (a copy)


class Probe(nn.Module):
    """Wraps a body and records how it was called. Reads nothing on the host."""

    def __init__(self, body):
        super().__init__()
        self.body = body
        self.calls: list[Call] = []

    def forward(self, x):
        kind = x.device.type
        logits = self.body(x)
        self.calls.append(
            Call(
                torch.is_autocast_enabled(kind),
                torch.get_autocast_dtype(kind),
                logits.detach().clone(),
            )
        )
        return logits


def fixed_linear(weight):
    """A bias-free linear layer with the given weight matrix (rows are vocabulary items)."""
    weight = torch.tensor(weight, dtype=torch.float32)
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    return Probe(layer)


def mlp(ctx, d_in, hidden, vocab, gain=3.0):
    """Linear, LayerNorm, GELU, Linear with seeded weights; logits of spread `gain`."""
    body = nn.Sequential(
        nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, vocab)
    )
    scales = [d_in**-0.5, 0.1, 0.1, 0.1, gain * hidden**-0.5, 0.5]
    with torch.no_grad():
        for param, scale in zip(body.parameters(), scales, strict=True):
            param.copy_(torch.from_numpy(ctx.randn(*param.shape)) * scale)
        body[1].weight.add_(1.0)
    return Probe(body)


# --------------------------------------------------------------------------------------
# Expected values and the checks every case makes
# --------------------------------------------------------------------------------------


def stats_float64(logits, temperature):
    """Log-probabilities and entropy of `logits / temperature`, in float64 NumPy."""
    z = logits.double().cpu().numpy() / temperature
    z = z - z.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    entropy = -(np.exp(log_probs) * log_probs).sum(axis=-1)
    return log_probs.astype(np.float32), entropy.astype(np.float32)


def name(dtype):
    return str(dtype).removeprefix("torch.")


def run(ctx, model, x, dtype, temperature=None, device=CPU):
    """Call the solution once; check how the model was called, then the numbers."""
    extra = () if temperature is None else (temperature,)
    out = ctx.solution.forward_amp(model, x, device, dtype, *extra)
    temperature = 1.0 if temperature is None else temperature

    # 1. How the body was run.
    check.that(
        len(model.calls) == 1,
        f"{FN}: called the model {len(model.calls)} times",
        "the model must be called exactly once",
    )
    call = model.calls[0]
    check.that(
        call.autocast,
        f"{FN}: the model ran with autocast disabled for its device ({device.type})",
        "The body must run inside torch.autocast(device_type=device.type, dtype=dtype). "
        "A region entered for another device type warns and turns itself off.",
    )
    check.that(
        call.region_dtype == dtype,
        f"{FN}: the autocast region used {name(call.region_dtype)}, expected {name(dtype)} "
        "(the requested dtype)",
        f"requested: {name(dtype)}",
    )
    check.that(
        call.logits.dtype == dtype,
        f"{FN}: the model's last layer computed in {name(call.logits.dtype)}, expected "
        f"{name(dtype)}",
        f"inside the region its linear layers must produce {name(dtype)}",
    )
    stray = sorted({name(p.dtype) for p in model.parameters()} - {"float32"})
    check.that(
        not stray,
        f"{FN}: the model's parameters are {', '.join(stray)} after the call; "
        "the model must not be modified",
    )

    # 2. What came back.
    check.that(
        isinstance(out, tuple) and len(out) == 2 and all(isinstance(t, torch.Tensor) for t in out),
        f"{FN}: must return a tuple (log_probs, entropy) of two tensors",
    )
    for label, tensor in zip(("log_probs", "entropy"), out, strict=True):
        check.that(
            tensor.dtype == torch.float32,
            f"{label}: got dtype {name(tensor.dtype)}, expected dtype float32",
            "the statistics are float32",
        )
        check.that(
            tensor.device.type == device.type,
            f"{label}: the result is on {tensor.device.type}, not on the model's device",
        )
    log_probs, entropy = (t.detach().cpu().numpy() for t in out)

    # 3. The numbers, from the logits this very call produced.
    want_log_probs, want_entropy = stats_float64(call.logits, temperature)
    check.allclose(log_probs, want_log_probs, rtol=1e-4, atol=1e-5, label="log_probs")
    check.allclose(entropy, want_entropy, rtol=1e-4, atol=1e-5, label="entropy")
    return log_probs, entropy, call


# --------------------------------------------------------------------------------------
# CPU cases
# --------------------------------------------------------------------------------------

EXAMPLE_WEIGHT = [[40.0, 0.0], [0.0, 40.0], [40.0, 40.0]]
EXAMPLE_X = [[1.5, 0.25]]  # logits [60, 10, 70], exact in float16 and bfloat16


@case("the worked example: logits [60, 10, 70] at temperature 20, bfloat16", public=True)
def worked_example(ctx):
    model = fixed_linear(EXAMPLE_WEIGHT)
    log_probs, entropy, call = run(ctx, model, torch.tensor(EXAMPLE_X), BF16, 20.0)
    if call.logits.float().tolist() != [[60.0, 10.0, 70.0]]:
        raise RuntimeError("authoring error: the example's logits changed")
    if not np.allclose(log_probs, [[-1.0046, -3.5046, -0.5046]], atol=1e-4) or not np.allclose(
        entropy, [0.7779], atol=1e-4
    ):
        raise RuntimeError("authoring error: the worked example changed")


@case("the same logits at temperature 0.001 in float16: 70000 does not fit", public=True)
def low_temperature(ctx):
    # Rejects scales_in_half and whole_pipeline_half: 70 / 0.001 is inf in float16,
    # and inf - inf is NaN. In float32 the answer is [-10000, -60000, 0], entropy 0.
    model = fixed_linear(EXAMPLE_WEIGHT)
    run(ctx, model, torch.tensor(EXAMPLE_X), F16, 0.001)
    model = fixed_linear([[300.0, 0.0], [0.0, 300.0], [-200.0, 100.0], [8.0, 8.0]])
    run(ctx, model, torch.tensor([[1.0, 0.5], [0.25, 1.0], [-1.0, -1.0]]), F16, 0.004)


@case("a small network in bfloat16: 16-bit body, float32 statistics", public=True)
def network_bfloat16(ctx):
    # Rejects no_autocast, hardcoded_cuda, converts_model (how the body ran),
    # returns_half (dtype), and whole_pipeline_half and entropy_in_half (numbers).
    model = mlp(ctx, 8, 32, 50)
    run(ctx, model, torch.from_numpy(ctx.randn(4, 8)), BF16)


@case("temperature is optional and defaults to 1", public=True)
def default_temperature(ctx):
    model = mlp(ctx, 8, 32, 50)
    x = torch.from_numpy(ctx.randn(4, 8))
    without, _, call = run(ctx, model, x, BF16)
    model.calls.clear()
    with_one, _, _ = run(ctx, model, x, BF16, 1.0)
    check.allclose(without, with_one, rtol=1e-6, atol=1e-7, label="log_probs")
    # At temperature 1 the log-probabilities are the logits minus a row constant.
    shift = call.logits.float().numpy() - without
    check.allclose(
        shift, np.broadcast_to(shift[:, :1], shift.shape), rtol=1e-4, atol=1e-4, label="log_probs"
    )


@case("the requested dtype is float16, on a CPU whose default would be bfloat16")
def float16_on_cpu(ctx):
    # Rejects ignores_dtype: without dtype= the CPU region is bfloat16.
    model = mlp(ctx, 8, 32, 50)
    run(ctx, model, torch.from_numpy(ctx.randn(4, 8)), F16, 0.8)


@case("temperatures that are not powers of two: the division happens in float32")
def odd_temperatures(ctx):
    # Rejects scales_in_half in bfloat16, where nothing overflows: a quotient rounded
    # to 8 significant bits is off by up to 0.4 percent.
    for temperature in (0.7, 1.3, 0.35, 3.0):
        model = mlp(ctx, 8, 32, 50)
        run(ctx, model, torch.from_numpy(ctx.randn(4, 8)), BF16, temperature)


@case("a vocabulary of 4096: the entropy is a sum of thousands of small terms")
def large_vocabulary(ctx):
    # Rejects entropy_in_half: in bfloat16 a term below 1/256 of the running sum is lost.
    model = mlp(ctx, 8, 32, 4096, gain=1.5)
    _, entropy, _ = run(ctx, model, torch.from_numpy(ctx.randn(4, 8)), BF16)
    check.that(
        bool(np.all(entropy > 5.0)),
        f"{FN}: authoring check failed, the distributions are not flat enough",
    )
    model = mlp(ctx, 8, 32, 4096, gain=1.5)
    run(ctx, model, torch.from_numpy(ctx.randn(2, 8)), F16, 2.0)


@case("mixed precision agrees with a float32 forward pass to within 0.25")
def close_to_float32(ctx):
    # The stated accuracy of the 16-bit body: the log-probabilities of the tokens that
    # matter (probability above 1e-4) are within 0.25 of a full float32 forward pass.
    for dtype, bound in ((BF16, 0.25), (F16, 0.05)):
        model = mlp(ctx, 8, 32, 50)
        x = torch.from_numpy(ctx.randn(6, 8))
        log_probs, _, _ = run(ctx, model, x, dtype)
        with torch.no_grad():
            full, _ = stats_float64(model.body(x), 1.0)
        likely = full > math.log(1e-4)
        worst = float(np.abs(log_probs - full)[likely].max())
        ctx.metric(f"max_abs_error_{name(dtype)}", worst)
        check.that(
            worst <= bound,
            f"log_probs: differ from a float32 forward pass by {worst:.3g}, more than a "
            f"{name(dtype)} body explains",
            f"bound for {name(dtype)}: {bound}",
        )


@case("one row, two tokens; and a batch of sixteen")
def edge_shapes(ctx):
    model = fixed_linear([[2.0], [-2.0]])
    log_probs, entropy, _ = run(ctx, model, torch.tensor([[0.0]]), BF16, 0.5)
    check.allclose(log_probs, [[-math.log(2.0)] * 2], label="log_probs")
    check.allclose(entropy, [math.log(2.0)], label="entropy")
    model = mlp(ctx, 4, 16, 7)
    run(ctx, model, torch.from_numpy(ctx.randn(16, 4)), F16, 1.7)


@case("random networks, dtypes, and temperatures: every row is a distribution")
def random_sweep(ctx):
    shapes = [(1, 4, 16, 5), (3, 8, 32, 50), (5, 6, 24, 300)]

    def prop(rng):
        batch, d_in, hidden, vocab = shapes[int(rng.integers(0, len(shapes)))]
        dtype = (BF16, F16)[int(rng.integers(0, 2))]
        temperature = float(rng.choice([0.25, 0.6, 1.0, 1.9, 5.0]))
        model = mlp(ctx, d_in, hidden, vocab, gain=float(rng.uniform(0.5, 4.0)))
        x = torch.from_numpy(rng.standard_normal((batch, d_in)).astype(np.float32))
        log_probs, entropy, _ = run(ctx, model, x, dtype, temperature)
        total = np.exp(log_probs.astype(np.float64)).sum(axis=-1)
        check.allclose(total, np.ones(batch), rtol=1e-4, label="sum of probabilities")
        check.that(
            bool(np.all(entropy >= -1e-5) and np.all(entropy <= math.log(vocab) + 1e-4)),
            "entropy: a value is outside the range an entropy can take",
            f"must be between 0 and log(V) = {math.log(vocab):.4f}",
        )

    check.forall(prop, trials=12)


# --------------------------------------------------------------------------------------
# GPU cases. Never run on the authoring machine; written from the torch 2.15 docs.
# --------------------------------------------------------------------------------------


@case("on the GPU: a float16 body, float32 statistics on the CUDA device", requires="cuda")
def gpu_float16(ctx):
    device = torch.device("cuda")
    for temperature in (1.0, 0.7):
        model = mlp(ctx, 8, 32, 50).to(device)
        x = torch.from_numpy(ctx.randn(4, 8)).to(device)
        run(ctx, model, x, F16, temperature, device)


@case("on the GPU: temperature 0.001 with logits that overflow float16", requires="cuda")
def gpu_low_temperature(ctx):
    device = torch.device("cuda")
    model = fixed_linear(EXAMPLE_WEIGHT).to(device)
    log_probs, entropy, _ = run(ctx, model, torch.tensor(EXAMPLE_X, device=device), F16, 0.001)
    check.finite(log_probs, label="log_probs")
    check.finite(entropy, label="entropy")
