# BUG: packs only complete groups of 8 and drops the remaining n % 8 codes instead of padding them with zeros into one more word.
import torch

SHIFTS = 4 * torch.arange(8)


def pack4(codes: torch.Tensor) -> torch.Tensor:
    complete = codes[: codes.shape[0] // 8 * 8].to(torch.int64)
    words = (complete.reshape(-1, 8) << SHIFTS).sum(dim=1)
    return words.to(torch.uint32)


def unpack4(words: torch.Tensor, n: int) -> torch.Tensor:
    nibbles = (words.to(torch.int64)[:, None] >> SHIFTS) & 0xF
    return nibbles.reshape(-1)[:n].to(torch.uint8)
