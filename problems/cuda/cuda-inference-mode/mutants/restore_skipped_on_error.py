# BUG: restores the training flags after the forward pass without try/finally; when the model raises, it is left in evaluation mode.
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    with torch.inference_mode():
        out = model(x.to(device))
    for module, training in flags:
        module.training = training
    return out


def as_inference(fn: Callable) -> Callable:
    return torch.inference_mode()(fn)
