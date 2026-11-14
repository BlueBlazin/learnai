# BUG: fills each byte high nibble first (code 0 in bits 4-7, code 1 in bits 0-3), the order hex digits are written in, while the bytes themselves are in the right order.
import torch

SHIFTS = torch.tensor([4, 0, 12, 8, 20, 16, 28, 24])


def pack4(codes: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(codes.to(torch.int64), (0, -codes.shape[0] % 8))
    words = (padded.reshape(-1, 8) << SHIFTS).sum(dim=1)
    return words.to(torch.uint32)


def unpack4(words: torch.Tensor, n: int) -> torch.Tensor:
    nibbles = (words.to(torch.int64)[:, None] >> SHIFTS) & 0xF
    return nibbles.reshape(-1)[:n].to(torch.uint8)
