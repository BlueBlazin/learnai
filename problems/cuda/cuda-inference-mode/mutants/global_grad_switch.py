# BUG: turns autograd off with torch.set_grad_enabled(False) as a statement and never turns it back on; the caller is left with gradients disabled, and the result is not an inference tensor.
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    torch.set_grad_enabled(False)
    try:
        return model(x.to(device))
    finally:
        for module, training in flags:
            module.training = training


def as_inference(fn: Callable) -> Callable:
    return torch.inference_mode()(fn)
