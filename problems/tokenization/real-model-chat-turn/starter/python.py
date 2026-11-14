from collections.abc import Callable, Iterator

import numpy as np

from whetstone.kit import ByteLevelBPE

START_OF_ROLE = "<|start_of_role|>"
END_OF_ROLE = "<|end_of_role|>"
END_OF_TEXT = "<|end_of_text|>"
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
)


def chat_turn(
    messages: list[dict[str, str]],
    tokenizer: ByteLevelBPE,
    step: Callable[[list[int]], np.ndarray],
    max_new_tokens: int,
) -> Iterator[str]:
    """Generate the assistant's reply to `messages`, yielding its text piece by piece.

    messages: [{"role": "system" | "user" | "assistant", "content": str}, ...], not empty.
    tokenizer: tokenizer.encode(text) -> list[int] (added tokens in the text become their
        ids), tokenizer.token_bytes(token_id) -> bytes, tokenizer.specials: dict from the
        text of each special token to its id.
    step: step(token_ids) feeds a non-empty list of ints to the model, after everything
        fed before, and returns the logits of the next token: float32 numpy, shape (V,).
        Call it once with the whole prompt, then once per generated token.
    max_new_tokens: at most this many tokens are generated, >= 1.
    Yields non-empty strings: the reply's text, each piece as soon as it is final.
    """
    raise NotImplementedError
    yield ""  # makes this function a generator, like the solution must be
