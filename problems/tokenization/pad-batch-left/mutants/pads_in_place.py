# BUG: pads the caller's lists in place and returns them as the rows of ids, so the input sequences are modified (and a second call sees already-padded input).
def pad_batch(
    seqs: list[list[int]], pad_id: int, side: str = "left"
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    if side not in ("left", "right"):
        raise ValueError(f'side must be "left" or "right", got {side!r}')
    if not seqs:
        raise ValueError("cannot pad an empty batch")
    lengths = [len(seq) for seq in seqs]
    if min(lengths) == 0:
        raise ValueError("every sequence needs at least one token")
    width = max(lengths)
    attention_mask = []
    for seq, n in zip(seqs, lengths, strict=True):
        pads, zeros = [pad_id] * (width - n), [0] * (width - n)
        ones = [1] * n
        if side == "left":
            seq[:0] = pads
            attention_mask.append(zeros + ones)
        else:
            seq.extend(pads)
            attention_mask.append(ones + zeros)
    return seqs, attention_mask, lengths
