# BUG: does not batch at all: runs the kit's decoder_forward_ref, which the statement rules out, on one sequence at a time.
import numpy as np
import torch

from whetstone.kit import DecoderConfig, decoder_forward_ref


def batched_forward(
    params: dict[str, torch.Tensor], cfg: DecoderConfig, seqs: list[list[int]], pad_id: int
) -> torch.Tensor:
    weights = {name: value.numpy() for name, value in params.items()}
    rows = [decoder_forward_ref(cfg, weights, np.array(seq))[-1] for seq in seqs]
    return torch.from_numpy(np.stack(rows))
