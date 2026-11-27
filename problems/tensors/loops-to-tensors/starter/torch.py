import torch


def segment_means(x: torch.Tensor, seg: torch.Tensor, S: int) -> torch.Tensor:
    """The mean of the rows of every segment. No Python loop over the rows.

    x: float32, shape (N, D).
    seg: int64, shape (N,), with 0 <= seg[n] < S: the segment of row n.
    S: the number of segments, a positive Python int.
    Returns float32, shape (S, D): row s is the mean of the rows n with seg[n] == s,
    and all zeros when there is no such row.
    """
    raise NotImplementedError


def common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """How many leading positions of each row pair agree. No Python loop over the rows.

    a, b: int64, shape (B, T).
    Returns an integer tensor, shape (B,): the largest n with a[i, :n] == b[i, :n]
    everywhere, which is T for equal rows.
    """
    raise NotImplementedError


def repeated_bigram_ban(ids: torch.Tensor, V: int) -> torch.Tensor:
    """The tokens that would repeat a pair already in the row. No Python loop over the rows.

    ids: int64, shape (B, T), with 0 <= ids[b, t] < V.
    V: the vocabulary size, a positive Python int.
    Returns bool, shape (B, V): out[b, v] is True when some t <= T - 2 has
    ids[b, t] == ids[b, T - 1] and ids[b, t + 1] == v.
    """
    raise NotImplementedError
