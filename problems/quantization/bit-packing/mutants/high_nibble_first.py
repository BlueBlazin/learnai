# BUG: puts the first code of each group in the highest four bits (big-endian nibble order) instead of the lowest; pack4 and unpack4 agree with each other but not with the layout.
import torch

SHIFTS = 28 - 4 * torch.arange(8)


def pack4(codes: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(codes.to(torch.int64), (0, -codes.shape[0] % 8))
    words = (padded.reshape(-1, 8) << SHIFTS).sum(dim=1)
    return words.to(torch.uint32)


def unpack4(words: torch.Tensor, n: int) -> torch.Tensor:
    nibbles = (words.to(torch.int64)[:, None] >> SHIFTS) & 0xF
    return nibbles.reshape(-1)[:n].to(torch.uint8)
