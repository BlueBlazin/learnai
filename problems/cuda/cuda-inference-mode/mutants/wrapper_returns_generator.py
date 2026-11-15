# BUG: as_inference wraps the call in a with block; for a generator function the block ends before the body starts, so every step runs with autograd on.
import functools
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
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return fn(*args, **kwargs)

    return wrapper
