# BUG: as_inference keeps the with block open across yields of a generator; while the generator is suspended the caller's own code runs in inference mode.
import functools
import inspect
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            return model(x.to(device))
    finally:
        for module, training in flags:
            module.training = training


def as_inference(fn: Callable) -> Callable:
    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def generator_wrapper(*args, **kwargs):
            with torch.inference_mode():
                yield from fn(*args, **kwargs)

        return generator_wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)

    return wrapper
