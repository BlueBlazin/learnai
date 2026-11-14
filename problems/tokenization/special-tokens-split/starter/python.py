from collections.abc import Callable


def encode_with_specials(
    text: str,
    specials: dict[str, int],
    encode_ordinary: Callable[[str], list[int]],
    allowed: set[str] | str,
) -> list[int]:
    """Encode `text`, turning special-token strings into their ids only where allowed.

    text: the string to encode; may be empty.
    specials: every special token of the tokenizer, string -> id.
    encode_ordinary: encodes text that contains no special tokens (it would encode a
        special string character by character, like any other text).
    allowed: a set of special strings (a subset of the keys of `specials`) that may be
        turned into their ids, or the string "none_as_text".
    Returns the token ids: `text` is split at special strings (earliest position first,
    longest string at that position), each special becomes its id, and each run of text
    between them is encoded with one call of `encode_ordinary`.
    Raises ValueError if a special string that is not in `allowed` is found.
    With allowed == "none_as_text", returns encode_ordinary(text): nothing is special.
    """
    raise NotImplementedError
