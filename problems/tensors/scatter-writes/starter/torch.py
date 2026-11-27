import torch


def one_hot_rows(ids: torch.Tensor, V: int) -> torch.Tensor:
    """One row of zeros per id, with a 1 at that id's column.

    ids: int64, shape (B,), with 0 <= ids[b] < V.
    V: the number of columns, a positive Python int.
    Returns float32, shape (B, V), with out[b, ids[b]] = 1 and 0 elsewhere.
    Do not call the framework's built-in one_hot.
    """
    raise NotImplementedError


def token_counts(ids: torch.Tensor, V: int) -> torch.Tensor:
    """How many times each token occurs in each row.

    ids: int64, shape (B, T), with 0 <= ids[b, t] < V.
    V: the vocabulary size, a positive Python int.
    Returns an integer tensor, shape (B, V); out[b, v] is the number of t with
    ids[b, t] == v, so a repeated token counts every time.
    """
    raise NotImplementedError


def write_at(x: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """A copy of x with K entries of every row replaced.

    x: float32, shape (B, V). It must not be changed.
    idx: int64, shape (B, K), with 0 <= idx[b, k] < V and no repeats within a row.
    values: float32, shape (B, K).
    Returns float32, shape (B, V), with out[b, idx[b, k]] = values[b, k] and
    out[b, v] = x[b, v] for every column v that row b of idx does not name.
    """
    raise NotImplementedError
