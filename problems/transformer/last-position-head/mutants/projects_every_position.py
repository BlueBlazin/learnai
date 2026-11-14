# BUG: projects all T positions to the vocabulary and then keeps the last row; the numbers are right, but it does T times the work and builds a (B, T, V) tensor to throw away all but one row per sequence.
import torch


def next_token_logits(hidden: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    return (hidden @ w_out.T)[:, -1]
