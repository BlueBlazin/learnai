If you build the prompt string first and then search it for `<|im_start|>`, what happens
when a user's message contains `<|im_start|>`? At what point in the rendering process do
you still know for sure whether a string is a template special token or user text?

---

Build the segment list as you render, then derive the final string at the end with
`"".join(...)`. Use two small helpers to keep the list canonical. `special(token)`
appends `("special", token)`. `text(piece)` ignores empty strings. It merges `piece`
into the previous segment if that segment is already `"text"`, and otherwise appends
`("text", piece)`. With those helpers, the function directly follows the four parts of
the specification: BOS, the default system turn when the first message is not
`"system"`, the messages, and the optional generation prompt.

---

The only stateful part is grouping consecutive `"tool"` messages, and checking the
previous and next message is enough:

```python
if role == "tool":
    if i == 0 or messages[i - 1]["role"] != "tool":      # the run starts here
        special(IM_START)
        text("user")
    text("\n<tool_response>\n" + content + "\n</tool_response>")
    if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":   # and ends here
        special(IM_END)
        text("\n")
```

Every other message emits `special(IM_START)`, `text(role + "\n" + content)`,
`special(IM_END)`, and `text("\n")`. The generation prompt emits `special(IM_START)` and
`text("assistant\n")`.
