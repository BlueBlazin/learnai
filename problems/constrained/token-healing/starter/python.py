from whetstone.kit import ToyVocab


def heal(prompt_ids: list[int], vocab: ToyVocab) -> tuple[list[int], bytes]:
    """Take the unstable tail off a prompt.

    prompt_ids: the prompt's token ids (possibly empty).
    vocab: the token table; vocab.bytes(t) is the byte string of token t (b"" for a
        special token), vocab.is_special(t) says whether t is special, vocab.size is
        the number of tokens.
    Returns (trimmed_ids, required_prefix): the prompt without its longest healable
    suffix (a list of ints), and the concatenated bytes of the removed tokens. A
    suffix is healable when it contains no special token and some vocabulary token
    starts with the suffix's bytes without being the suffix's only token. With no
    healable suffix: (the prompt unchanged, b"").
    """
    raise NotImplementedError


def healing_allowed(required_prefix: bytes, generated_bytes: bytes, vocab: ToyVocab) -> list[int]:
    """The tokens that may be generated next while the removed text is being restored.

    required_prefix: the bytes that generation must begin with (from `heal`).
    generated_bytes: the bytes of all tokens generated so far.
    Returns token ids in increasing order: every id once generated_bytes covers
    required_prefix; before that, the tokens with at least one byte whose bytes are a
    prefix of the part still missing, or which the missing part is a prefix of.
    Raises ValueError if neither argument is a prefix of the other.
    """
    raise NotImplementedError
