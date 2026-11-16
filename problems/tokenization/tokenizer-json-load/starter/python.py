import re
from collections.abc import Callable

from learnai.kit.bpe import translate_pattern


def load_tokenizer(
    tokenizer_json: dict,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Build (encode, decode) from a parsed Hugging Face tokenizer.json (Granite's layout).

    tokenizer_json: the parsed file. What matters in it:
        ["model"]["vocab"]: token string (bytes written in the ByteLevel alphabet) -> id
        ["model"]["merges"]: list of [left, right]; the position in the list is the rank
        ["added_tokens"]: list of {"content": str, "id": int, "special": bool, ...}
        ["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]: the word pattern;
            re.compile(translate_pattern(pattern)) is its Python equivalent
    Returns two functions:
        encode(text: str) -> list[int]: the ids the `tokenizers` library gives the text,
            with nothing added before or after
        decode(ids: list[int]) -> str: the text of the ids, every token kept
    """
    raise NotImplementedError
