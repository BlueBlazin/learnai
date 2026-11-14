def pad_batch(
    seqs: list[list[int]], pad_id: int, side: str = "left"
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Pad token sequences of different lengths into one rectangular batch.

    seqs: B sequences of token ids, each with at least one token.
    pad_id: the token id that fills the empty positions.
    side: "left" (pads first, tokens last; the default) or "right" (tokens first).
    Returns (ids, attention_mask, lengths):
        ids: B rows of T ints, T = the longest sequence's length.
        attention_mask: B rows of T ints; 1 where ids holds a token of the sequence,
            0 where it holds padding.
        lengths: B ints, the number of real tokens in each row.
    Raises ValueError if seqs is empty, if any sequence is empty, or if side is
    neither "left" nor "right".
    """
    raise NotImplementedError
