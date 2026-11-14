# BUG: implements nothing: hands the JSON to the tokenizers library, which the statement rules out.
import json
from collections.abc import Callable


def load_tokenizer(
    tokenizer_json: dict,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(json.dumps(tokenizer_json))

    def encode(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False).ids

    def decode(ids: list[int]) -> str:
        return tok.decode(ids, skip_special_tokens=False)

    return encode, decode
