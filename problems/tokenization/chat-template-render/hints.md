If you first build the string and then look for `<|im_start|>` in it to find the
special tokens, what happens when a user's message contains `<|im_start|>`? At what
moment do you still know for certain whether a piece of the prompt is a special token
or text?

---

Build the segment list as you go, and derive the string from it at the end with
`"".join(...)`. Two small helpers keep the list canonical: `special(token)` appends
`("special", token)`, and `text(piece)` appends text, except that it does nothing for
an empty piece and, when the last segment is already text, extends that segment
instead of adding a new one. Then the function is a transcription of the four parts of
the specification: BOS, the default system turn if the first message is not a system
message, the messages, the generation prompt.

---

The only part with state is the tool run. Looking at the neighbors is enough:

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

Every other message is `special(IM_START)`, `text(role + "\n" + content)`,
`special(IM_END)`, `text("\n")`, and the generation prompt is `special(IM_START)`,
`text("assistant\n")`.
