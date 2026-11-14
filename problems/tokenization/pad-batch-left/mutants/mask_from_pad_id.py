# BUG: the attention mask is computed by comparing the padded ids with pad_id, so a real token that happens to equal pad_id (common when the pad token is the EOS token) is masked out.
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
    ids = []
    for seq, n in zip(seqs, lengths, strict=True):
        pads = [pad_id] * (width - n)
        ids.append(pads + list(seq) if side == "left" else list(seq) + pads)
    attention_mask = [[int(token != pad_id) for token in row] for row in ids]
    return ids, attention_mask, lengths
