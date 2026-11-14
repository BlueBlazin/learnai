There are three separate jobs here, and each can be tested on its own: a pure
function from `messages` to the prompt *text*, a loop that turns logits into tokens
and decides when to stop, and a decoder that turns token bytes into text. Which of the
three decides what `step` is called with, and which decides what is yielded?

---

The prompt is a list of `(role, content)` turns: first `("system", text)`, where
`text` is the first message's content if that message is a non-empty system message
and the default otherwise; then every other message. Join
`<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n` over the turns and
append `<|start_of_role|>assistant<|end_of_role|>`. Encode the whole string once.

The loop: `logits = step(prompt_ids)`, then for `count = 1 .. max_new_tokens`: take
the argmax; stop if it is the id of `<|end_of_text|>`; produce its text; stop if
`count == max_new_tokens`; otherwise `logits = step([token])`. Note where the yield
sits: before the call of `step`.

---

For the text, keep one incremental decoder for the whole reply:

```python
decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
...
piece = decoder.decode(b"" if token in special_ids else tokenizer.token_bytes(token))
if piece:
    yield piece
...
tail = decoder.decode(b"", final=True)  # after the loop
if tail:
    yield tail
```

`special_ids` is `set(tokenizer.specials.values())`, and the stop id is
`tokenizer.specials["<|end_of_text|>"]`. Convert the argmax to a Python `int` before
passing it on: `step` wants a list of ints, and the ids of the real model do not fit
in 16 bits.
