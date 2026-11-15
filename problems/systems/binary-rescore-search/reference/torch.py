import torch

# How many bits are set in each possible byte: popcount by table lookup.
POPCOUNT = torch.tensor([byte.bit_count() for byte in range(256)])
# The value of each of the eight bits of a byte, the first bit most significant.
BIT_VALUES = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1])


def binarize(e: torch.Tensor) -> torch.Tensor:
    bits = (e > 0).reshape(*e.shape[:-1], -1, 8)  # (..., D / 8, 8): one row per byte
    return (bits * BIT_VALUES).sum(dim=-1).to(torch.uint8)


def hamming_topk(
    q_bits: torch.Tensor, corpus_bits: torch.Tensor, m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # XOR leaves a 1 wherever two codes differ; the distance is the number of 1 bits.
    differing = torch.bitwise_xor(corpus_bits, q_bits)  # (N, D / 8) uint8
    distances = POPCOUNT[differing.long()].sum(dim=-1)  # (N,)
    # A stable sort orders equal distances by index.
    order = torch.argsort(distances, stable=True)[:m]
    return order, distances[order]


def search(
    q: torch.Tensor,
    corpus_bits: torch.Tensor,
    corpus_int8: torch.Tensor,
    k: int,
    oversample: int,
    distance,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Stage 1: cheap and coarse. More candidates than needed, by Hamming distance.
    candidates, _ = hamming_topk(binarize(q), corpus_bits, k * oversample)
    # Stage 2: precise, on the candidates only. The query keeps its full precision.
    scores = distance(q, corpus_int8[candidates])
    order = torch.argsort(scores, descending=True, stable=True)[:k]
    return candidates[order], scores[order]
