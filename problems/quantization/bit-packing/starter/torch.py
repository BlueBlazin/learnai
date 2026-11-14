import torch


def pack4(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes into 32-bit words, eight per word, lowest index in the lowest bits.

    codes: uint8 tensor of shape (n,), n >= 1, every value in [0, 15].
    Returns a uint32 tensor of shape (ceil(n / 8),). Code 8k + i occupies bits 4i to
    4i + 3 of word k; positions past the end of codes hold 0.
    """
    raise NotImplementedError


def unpack4(words: torch.Tensor, n: int) -> torch.Tensor:
    """Recover the first n 4-bit codes from words packed by pack4.

    words: uint32 tensor of shape (m,). n: number of codes, 1 <= n <= 8 * m.
    Returns a uint8 tensor of shape (n,) with values in [0, 15].
    """
    raise NotImplementedError
