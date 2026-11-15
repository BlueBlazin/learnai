import torch


def binarize(e: torch.Tensor) -> torch.Tensor:
    """Sign bits of `e`, packed eight to a byte.

    e: float32, shape (..., D) with D a multiple of 8.
    Returns uint8, shape (..., D // 8). Bit j of a vector is 1 where e[..., j] > 0;
    byte i holds bits 8i to 8i + 7, the first of them in the most significant bit.
    """
    raise NotImplementedError


def hamming_topk(
    q_bits: torch.Tensor, corpus_bits: torch.Tensor, m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The m corpus codes nearest to the query code in Hamming distance.

    q_bits: uint8, shape (D // 8,). corpus_bits: uint8, shape (N, D // 8).
    Returns (indices, distances), both integer tensors of shape (min(m, N),), in
    ascending order of distance; equal distances are ordered by index, lower first.
    """
    raise NotImplementedError


def search(
    q: torch.Tensor,
    corpus_bits: torch.Tensor,
    corpus_int8: torch.Tensor,
    k: int,
    oversample: int,
    distance,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary retrieval of k * oversample candidates, then rescoring with `distance`.

    q: float32, shape (D,). corpus_bits: uint8, shape (N, D // 8).
    corpus_int8: int8, shape (N, D). distance: a kit CountingDistance with metric
    "dot"; distance(q, rows) returns the float32 inner products of q with the rows.
    Returns (indices, scores): the min(k, N) candidates with the highest scores, best
    first (integer and float32 tensors); equal scores keep the candidate order.
    """
    raise NotImplementedError
