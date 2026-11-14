# BUG: the default for side is "right" (the default of most tokenizer libraries) instead of "left".
def pad_batch(
    seqs: list[list[int]], pad_id: int, side: str = "right"
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    if side not in ("left", "right"):
        raise ValueError(f'side must be "left" or "right", got {side!r}')
    if not seqs:
        raise ValueError("cannot pad an empty batch")
    lengths = [len(seq) for seq in seqs]
    if min(lengths) == 0:
        # A row of nothing but padding has an all-zero mask: the model would have no
        # position to attend to and no last token to continue from.
        raise ValueError("every sequence needs at least one token")
    width = max(lengths)
    ids, attention_mask = [], []
    for seq, n in zip(seqs, lengths, strict=True):
        pads, zeros = [pad_id] * (width - n), [0] * (width - n)
        ones = [1] * n  # the mask comes from the length, never from comparing with pad_id
        if side == "left":
            ids.append(pads + list(seq))  # the last column is every row's last real token
            attention_mask.append(zeros + ones)
        else:
            ids.append(list(seq) + pads)
            attention_mask.append(ones + zeros)
    return ids, attention_mask, lengths
