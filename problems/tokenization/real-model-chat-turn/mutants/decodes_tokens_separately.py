# BUG: decodes each token's bytes on its own, so a character whose bytes are spread over several tokens arrives as replacement characters.
import codecs
from collections.abc import Callable, Iterator

import numpy as np

from whetstone.kit import ByteLevelBPE

START_OF_ROLE = "<|start_of_role|>"
END_OF_ROLE = "<|end_of_role|>"
END_OF_TEXT = "<|end_of_text|>"
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
)


def render_prompt(messages: list[dict[str, str]]) -> str:
    # The prompt always opens with a system turn. A leading system message supplies its
    # text (unless it is empty) and is not rendered a second time.
    system, rest = DEFAULT_SYSTEM, messages
    if messages[0]["role"] == "system":
        system, rest = messages[0]["content"] or DEFAULT_SYSTEM, messages[1:]
    turns = [("system", system)] + [(m["role"], m["content"]) for m in rest]
    text = "".join(
        f"{START_OF_ROLE}{role}{END_OF_ROLE}{content}{END_OF_TEXT}\n" for role, content in turns
    )
    # The generation prompt: an assistant turn left open, with nothing after the role.
    return text + f"{START_OF_ROLE}assistant{END_OF_ROLE}"


def chat_turn(
    messages: list[dict[str, str]],
    tokenizer: ByteLevelBPE,
    step: Callable[[list[int]], np.ndarray],
    max_new_tokens: int,
) -> Iterator[str]:
    stop = tokenizer.specials[END_OF_TEXT]
    silent = set(tokenizer.specials.values())  # special tokens are never shown as text
    # Holds back the bytes of an unfinished character, and nothing else.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    logits = step(tokenizer.encode(render_prompt(messages)))  # the whole prompt, once
    for count in range(1, max_new_tokens + 1):
        token = int(np.argmax(logits))
        if token == stop:
            break  # the end of the turn is neither text nor fed back
        data = b"" if token in silent else tokenizer.token_bytes(token)
        piece = data.decode("utf-8", errors="replace")
        if piece:
            yield piece  # before the next model call: the client sees it now
        if count == max_new_tokens:
            break  # the last token allowed: a further model call would be wasted
        logits = step([token])
    tail = decoder.decode(b"", final=True)  # a character that was never finished
    if tail:
        yield tail
