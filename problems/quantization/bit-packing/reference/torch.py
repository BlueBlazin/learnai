import torch

# Bit offset of each of the eight codes in a word: code i sits at bits 4i to 4i + 3.
SHIFTS = 4 * torch.arange(8)


def pack4(codes: torch.Tensor) -> torch.Tensor:
    # Widen before shifting: a uint8 shifted left stays uint8 and loses its high bits,
    # and PyTorch has no shifts for uint32. int64 holds every 32-bit word without a sign.
    padded = torch.nn.functional.pad(codes.to(torch.int64), (0, -codes.shape[0] % 8))
    # The eight shifted codes occupy disjoint bits, so adding them is the same as OR.
    words = (padded.reshape(-1, 8) << SHIFTS).sum(dim=1)
    return words.to(torch.uint32)


def unpack4(words: torch.Tensor, n: int) -> torch.Tensor:
    # uint32 -> int64 zero-extends, so the right shift brings in zeros, never sign bits.
    nibbles = (words.to(torch.int64)[:, None] >> SHIFTS) & 0xF
    # Row-major flattening restores the order; the padding codes are at the end.
    return nibbles.reshape(-1)[:n].to(torch.uint8)
