def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]:
    """Split one word into WordPiece token ids, greedy longest-match-first.

    word: a single word (no whitespace); may be empty; any Unicode characters.
    vocab: token string -> id. Word-initial pieces are stored as they are ("un"),
        continuation pieces with the prefix "##" ("##able").
    unk: the unknown token, always a key of vocab.
    max_chars: a word with more characters (code points) than this is not split.
    Returns the list of ids; [vocab[unk]] if the word is too long or if at some
    position no piece matches (the whole word, not only the failing part); [] for "".
    """
    raise NotImplementedError
