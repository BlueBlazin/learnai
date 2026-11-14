# BUG: does not run the forward pass at all: converts everything to numpy and calls the kit's decoder_forward_ref, which the statement rules out.
import torch

from whetstone.kit import DecoderConfig, decoder_forward_ref


def forward(params: dict[str, torch.Tensor], cfg: DecoderConfig, tokens: torch.Tensor) -> torch.Tensor:
    weights = {name: value.numpy() for name, value in params.items()}
    return torch.from_numpy(decoder_forward_ref(cfg, weights, tokens.numpy()))
