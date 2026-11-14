import mlx.core as mx


def pack4(codes: mx.array) -> mx.array:
    """Pack 4-bit codes into 32-bit words, eight per word, lowest index in the lowest bits.

    codes: uint8 array of shape (n,), n >= 1, every value in [0, 15].
    Returns a uint32 array of shape (ceil(n / 8),). Code 8k + i occupies bits 4i to
    4i + 3 of word k; positions past the end of codes hold 0.
    """
    raise NotImplementedError


def unpack4(words: mx.array, n: int) -> mx.array:
    """Recover the first n 4-bit codes from words packed by pack4.

    words: uint32 array of shape (m,). n: number of codes, 1 <= n <= 8 * m.
    Returns a uint8 array of shape (n,) with values in [0, 15].
    """
    raise NotImplementedError
