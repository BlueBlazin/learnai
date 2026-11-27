Split the problem into three independent pieces that you can test separately: a pure function from `messages` to the prompt text, a generation loop that turns logits into tokens and decides when to stop, and an incremental decoder that turns token bytes into text. Which piece decides what `step` is called with, and which piece decides what gets yielded?

---

Build the prompt as a list of `(role, content)` turns. Start with `("system", text)`, where `text` is `messages[0]["content"]` if the first message is a non-empty `"system"` message and the default system prompt otherwise. If `messages[0]` has role `"system"`, drop it from the remaining messages even when its content is empty. Otherwise keep all messages. Append every remaining message as its own `(role, content)` turn. Join `<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n` across all turns, append `<|start_of_role|>assistant<|end_of_role|>`, and encode the whole string once.

For the generation loop, call `logits = step(prompt_ids)` once. Then for `count` from `1` to `max_new_tokens`, take the argmax token and stop immediately if it is the ID of `<|end_of_text|>`. Otherwise decode and yield its text, stop if `count == max_new_tokens`, and only then call `logits = step([token])`. Make sure `yield` happens before `step([token])`.

---

Keep one incremental UTF-8 decoder for the whole reply:

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

Set `special_ids = set(tokenizer.specials.values())` and `stop = tokenizer.specials["<|end_of_text|>"]`. Convert the argmax result to a Python `int` before passing it to `step`, since `step` expects Python ints and Granite's token IDs exceed 16-bit integers.
