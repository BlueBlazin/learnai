# BUG: renders the prompt with a Jinja template engine instead of writing the layout out by hand, which the statement rules out.
import codecs
from collections.abc import Callable, Iterator

import jinja2
import numpy as np

from learnai.kit import ByteLevelBPE

END_OF_TEXT = "<|end_of_text|>"
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
)
TEMPLATE = (
    "{%- set first = messages[0] %}"
    "{%- set system = first.content if first.role == 'system' and first.content else default %}"
    "{{- '<|start_of_role|>system<|end_of_role|>' + system + '<|end_of_text|>\n' }}"
    "{%- for m in messages %}{%- if not (loop.first and m.role == 'system') %}"
    "{{- '<|start_of_role|>' + m.role + '<|end_of_role|>' + m.content + '<|end_of_text|>\n' }}"
    "{%- endif %}{%- endfor %}"
    "{{- '<|start_of_role|>assistant<|end_of_role|>' }}"
)


def chat_turn(
    messages: list[dict[str, str]],
    tokenizer: ByteLevelBPE,
    step: Callable[[list[int]], np.ndarray],
    max_new_tokens: int,
) -> Iterator[str]:
    stop = tokenizer.specials[END_OF_TEXT]
    silent = set(tokenizer.specials.values())
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = jinja2.Template(TEMPLATE).render(messages=messages, default=DEFAULT_SYSTEM)
    logits = step(tokenizer.encode(text))
    for count in range(1, max_new_tokens + 1):
        token = int(np.argmax(logits))
        if token == stop:
            break
        piece = decoder.decode(b"" if token in silent else tokenizer.token_bytes(token))
        if piece:
            yield piece
        if count == max_new_tokens:
            break
        logits = step([token])
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail
