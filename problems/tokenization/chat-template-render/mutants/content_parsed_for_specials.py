# BUG: builds the string first and derives the segments by searching it for special-token strings, so '<|im_end|>' typed by a user inside message content is marked as a special token.
import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SYSTEM = "You are a helpful assistant."


def render_chat(
    messages: list[dict[str, str]], add_generation_prompt: bool, bos: str | None
) -> tuple[str, list[tuple[str, str]]]:
    out = bos or ""
    if not messages or messages[0]["role"] != "system":
        out += f"{IM_START}system\n{DEFAULT_SYSTEM}{IM_END}\n"
    for i, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if role == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                out += IM_START + "user"
            out += f"\n<tool_response>\n{content}\n</tool_response>"
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                out += IM_END + "\n"
        else:
            out += f"{IM_START}{role}\n{content}{IM_END}\n"
    if add_generation_prompt:
        out += IM_START + "assistant\n"

    specials = [IM_START, IM_END] + ([bos] if bos is not None else [])
    pattern = "(" + "|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)) + ")"
    segments = [
        ("special" if piece in specials else "text", piece)
        for piece in re.split(pattern, out)
        if piece
    ]
    return out, segments
