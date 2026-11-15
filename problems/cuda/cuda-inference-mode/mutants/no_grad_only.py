# BUG: uses torch.no_grad() where inference mode is required; the result is an ordinary tensor that still carries a version counter and view tracking.
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            return model(x.to(device))
    finally:
        for module, training in flags:
            module.training = training


def as_inference(fn: Callable) -> Callable:
    return torch.inference_mode()(fn)
