# BUG: every tool message gets a user turn of its own; consecutive tool results are not grouped into one turn.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SYSTEM = "You are a helpful assistant."


def render_chat(
    messages: list[dict[str, str]], add_generation_prompt: bool, bos: str | None
) -> tuple[str, list[tuple[str, str]]]:
    segments: list[tuple[str, str]] = []

    def special(token: str) -> None:
        segments.append(("special", token))

    def text(piece: str) -> None:
        # Text is never inspected, only appended: merged into the previous text
        # segment if there is one, and dropped if it is empty.
        if not piece:
            return
        if segments and segments[-1][0] == "text":
            segments[-1] = ("text", segments[-1][1] + piece)
        else:
            segments.append(("text", piece))

    if bos is not None:
        special(bos)  # exactly once, before everything else
    if not messages or messages[0]["role"] != "system":
        special(IM_START)
        text("system\n" + DEFAULT_SYSTEM)
        special(IM_END)
        text("\n")
    for i, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if role == "tool":
            # Tool results are shown to the model inside a user turn; consecutive
            # results share one turn.
            special(IM_START)
            text("user")
            text("\n<tool_response>\n")
            text(content)
            text("\n</tool_response>")
            special(IM_END)
            text("\n")
        else:
            special(IM_START)
            text(role + "\n")
            text(content)
            special(IM_END)
            text("\n")
    if add_generation_prompt:
        special(IM_START)
        text("assistant\n")  # the turn is left open: the model writes the rest
    return "".join(piece for _, piece in segments), segments
