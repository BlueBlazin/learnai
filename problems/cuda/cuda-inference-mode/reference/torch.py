from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Remember every submodule's flag. `model.train(flag)` would write one value into
    # all of them and thaw a submodule that was put in evaluation mode on purpose.
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        # Inference mode records no graph, no version counters, and no view tracking;
        # the context manager restores the caller's mode even when the model raises.
        with torch.inference_mode():
            return model(x.to(device))
    finally:
        for module, training in flags:
            module.training = training


def as_inference(fn: Callable) -> Callable:
    # As a decorator, inference_mode re-enters the mode around every resumption of a
    # generator function and leaves it at every yield. A hand-written
    # `with torch.inference_mode(): return fn(...)` would return an unstarted
    # generator whose body later runs with autograd on.
    return torch.inference_mode()(fn)
