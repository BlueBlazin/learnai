# BUG: never moves x to the device; it works only when x already lives where the model does.
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            return model(x)
    finally:
        for module, training in flags:
            module.training = training


def as_inference(fn: Callable) -> Callable:
    return torch.inference_mode()(fn)
