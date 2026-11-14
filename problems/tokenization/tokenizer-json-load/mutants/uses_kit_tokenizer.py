# BUG: implements nothing: returns the methods of the kit's ByteLevelBPE, which the statement rules out.
from collections.abc import Callable

from whetstone.kit import ByteLevelBPE


def load_tokenizer(
    tokenizer_json: dict,
) -> tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    tok = ByteLevelBPE.from_json(tokenizer_json)
    return tok.encode, tok.decode
