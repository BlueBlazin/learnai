# BUG: unpacks from signed int32 words and does not mask the top nibble ("nothing lies above it"); the arithmetic right shift sign-extends, so a top code of 8 or more comes out as 248 or more.
import torch

SHIFTS = 4 * torch.arange(8)


def pack4(codes: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(codes.to(torch.int64), (0, -codes.shape[0] % 8))
    words = (padded.reshape(-1, 8) << SHIFTS).sum(dim=1)
    return words.to(torch.uint32)


def unpack4(words: torch.Tensor, n: int) -> torch.Tensor:
    signed = words.to(torch.int32)[:, None]
    low = (signed >> SHIFTS[:7]) & 0xF
    top = signed >> 28
    nibbles = torch.cat([low, top], dim=1)
    return nibbles.reshape(-1)[:n].to(torch.uint8)
