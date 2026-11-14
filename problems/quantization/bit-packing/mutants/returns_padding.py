# BUG: unpack4 ignores n and returns all 8 codes of every word, including the padding codes of the last word.
import torch

SHIFTS = 4 * torch.arange(8)


def pack4(codes: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(codes.to(torch.int64), (0, -codes.shape[0] % 8))
    words = (padded.reshape(-1, 8) << SHIFTS).sum(dim=1)
    return words.to(torch.uint32)


def unpack4(words: torch.Tensor, n: int) -> torch.Tensor:
    nibbles = (words.to(torch.int64)[:, None] >> SHIFTS) & 0xF
    return nibbles.reshape(-1).to(torch.uint8)
