# BUG: shifts the uint8 codes without widening them first; a uint8 shifted by a Python int stays uint8, so every bit pushed past bit 7 is lost and only two codes per word survive.
import torch


def pack4(codes: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(codes, (0, -codes.shape[0] % 8)).reshape(-1, 8)
    words = torch.zeros(padded.shape[0], dtype=torch.int64)
    for i in range(8):
        words |= padded[:, i] << (4 * i)
    return words.to(torch.uint32)


def unpack4(words: torch.Tensor, n: int) -> torch.Tensor:
    shifts = 4 * torch.arange(8)
    nibbles = (words.to(torch.int64)[:, None] >> shifts) & 0xF
    return nibbles.reshape(-1)[:n].to(torch.uint8)
