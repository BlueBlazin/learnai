UNK_PENALTY = 10.0


def unigram_encode(text: str, pieces: dict[str, float]) -> list[str]:
    """Split `text` into the highest-scoring sequence of pieces.

    text: any Python string (may be empty). Positions are characters (code points).
    pieces: the vocabulary, a non-empty dict from piece (a non-empty string) to its
        score, the log-probability of the piece.
    A character that is not itself a key of `pieces` may be emitted as a one-character
    unknown piece with the score min(pieces.values()) - UNK_PENALTY.
    Returns the list of pieces, which concatenate to `text`: the segmentation with the
    highest total score; among those, the one with the fewest pieces; among those, the
    one whose first piece is longest, then whose second piece is longest, and so on.
    """
    raise NotImplementedError
