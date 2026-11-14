import numpy as np
import torch

from whetstone.judge import case, check


@case("scales on the CPU", public=True)
def cpu(ctx):
    ctx.check("scale", np.array([1.0, -2.0, 0.5], dtype=np.float32), 3.0, torch.device("cpu"))


@case("random arrays on the CPU")
def cpu_random(ctx):
    ctx.check("scale", ctx.randn(4, 5), -1.5, torch.device("cpu"))


# Rejects stays_on_cpu, which only a machine with CUDA can tell apart.
@case("the result is on the CUDA device", requires="cuda")
def on_cuda(ctx):
    out = ctx.solution.scale(torch.ones(3), 2.0, torch.device("cuda"))
    check.that(out.device.type == "cuda", f"scale: the result is on {out.device.type}")
