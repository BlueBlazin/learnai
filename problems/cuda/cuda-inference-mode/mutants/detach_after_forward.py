# BUG: runs the forward pass with autograd recording and detaches the result; the whole graph and every activation are kept alive during the pass, and the result is not an inference tensor.
from collections.abc import Callable

import torch
from torch import nn


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        return model(x.to(device)).detach()
    finally:
        for module, training in flags:
            module.training = training


def as_inference(fn: Callable) -> Callable:
    return torch.inference_mode()(fn)
