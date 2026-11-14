IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SYSTEM = "You are a helpful assistant."


def render_chat(
    messages: list[dict[str, str]], add_generation_prompt: bool, bos: str | None
) -> tuple[str, list[tuple[str, str]]]:
    """Render a conversation in the ChatML layout.

    messages: dicts with the keys "role" ("system", "user", "assistant", or "tool")
        and "content" (a string, used verbatim).
    add_generation_prompt: if true, end with the opening of an assistant turn.
    bos: the model's beginning-of-sequence token string, or None if it has none.
    Returns (text, segments):
        text: the rendered prompt.
        segments: the same prompt as a list of (kind, string) tuples that concatenate
            to `text`. kind is "special" for one special token written by the template
            (bos, <|im_start|>, <|im_end|>) and "text" for everything else. Adjacent
            text is one segment, and no segment is empty. Message content is always
            text, whatever it contains.
    """
    raise NotImplementedError
