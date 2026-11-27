## The idea

Break the turn into three independent components executed in a fixed order.

**Prompt formatting is model-specific data.** Granite 4.0 ships a hundred-line Jinja template, which reduces to three rules for plain messages. Every prompt begins with a system turn. A leading non-empty system message replaces the default text rather than adding a second system turn, while an empty leading system message is ignored. Finally, the prompt ends with an open assistant header (`<|start_of_role|>assistant<|end_of_role|>`) and no trailing newline. Once the `(role, content)` pairs are assembled, formatting is a single string join. Encoding the full string at once is safe here because message contents are guaranteed to contain no special-token text. With untrusted user input, you would encode special tokens and message contents separately, as in `chat-template-render`.

**The generation loop has two exits that differ by one `step` call.** Because calling the model is the expensive operation, only call `step` when you will actually choose another token from its logits. Neither the `<|end_of_text|>` token nor the `max_new_tokens`-th generated token is followed by another token, so neither is fed to `step`. Total calls equal one for the prompt plus one per generated token, minus one if `max_new_tokens` ends the turn.

**Streaming text is a stateful byte stream.** Each token contributes a sequence of bytes, and a multi-byte UTF-8 character can cross token boundaries. An incremental UTF-8 decoder buffers only the trailing incomplete character (at most three bytes) and immediately emits everything finalized before it. Special tokens contribute `b""` so their names never leak into user-visible text. Flushing with `final=True` after the loop turns any trailing incomplete sequence into a single U+FFFD, guaranteeing that concatenating all yielded pieces matches decoding the full reply at once.

Yielding each piece before calling `step([token])` comes naturally in a generator. The caller receives the newly decoded text immediately, and the forward pass for the next token starts only when the caller asks for the next item.

## Reference solution (Python)

```python
def render_prompt(messages):
    system, rest = DEFAULT_SYSTEM, messages
    if messages[0]["role"] == "system":
        system, rest = messages[0]["content"] or DEFAULT_SYSTEM, messages[1:]
    turns = [("system", system)] + [(m["role"], m["content"]) for m in rest]
    text = "".join(
        f"{START_OF_ROLE}{role}{END_OF_ROLE}{content}{END_OF_TEXT}\n" for role, content in turns
    )
    return text + f"{START_OF_ROLE}assistant{END_OF_ROLE}"


def chat_turn(messages, tokenizer, step, max_new_tokens):
    stop = tokenizer.specials[END_OF_TEXT]
    silent = set(tokenizer.specials.values())
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    logits = step(tokenizer.encode(render_prompt(messages)))
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
```

## Common mistakes

- **Omitting the generation prompt.** Ending the prompt right after the user's `<|end_of_text|>\n` leaves the model free to start any role, often generating another user turn.
- **Adding a newline after the assistant header.** ChatML puts a newline there, but Granite does not. That single extra token puts every reply on a prefix the model never saw during training.
- **Mishandling the leading system turn.** Omitting the default system turn, rendering the first system message twice, or keeping an empty leading system message as an empty turn all violate the rule that every prompt starts with exactly one non-empty system turn.
- **Stripping message content.** The template inserts every `"content"` string verbatim.
- **Emitting the stop token or stopping on any special token.** Printing `<|end_of_text|>` at the end of a reply leaks control markers, while stopping on any special token cuts replies short whenever the model emits a non-stop special token mid-turn.
- **Rendering special tokens as text.** `tokenizer.token_bytes` returns the literal marker string for a special token, which is useful for debugging but should not appear in the reply.
- **Decoding token by token.** Calling `token_bytes(t).decode(errors="replace")` on each token turns multi-token emojis and many CJK characters into `�`.
- **Forgetting the final flush.** If `max_new_tokens` cuts off a multi-byte character mid-sequence, skipping `decode(b"", final=True)` silently drops it instead of emitting `�`, causing the streamed output to disagree with full-reply decoding.
- **Yielding empty strings.** Yielding a piece on every token forces downstream consumers to filter out empty chunks.
- **Off-by-one loop exits.** Generating one token too many or calling `step` after the `max_new_tokens`-th token breaks exact step counts and wastes a forward pass per request.
- **Prefilling the prompt one token at a time.** Passing prompt IDs individually triggers dozens of forward passes instead of a single prefill call.
- **Buffering the whole reply.** Collecting all pieces in a list and yielding at the end produces the right transcript but breaks streaming, turning time to first token into time to last token.
- **Storing token IDs in 16-bit integers.** Unsigned 16-bit integers fit the miniature tokenizer's IDs but overflow on Granite's first prompt token (100,264), since modern vocabularies exceed 65,536 tokens.

## Complexity

For a prompt of $P$ tokens, `chat_turn` makes one prefill call over $P$ tokens and at most `max_new_tokens - 1` single-token decode calls (or as many single-token calls as generated tokens when stopped early by `<|end_of_text|>`). With the model's KV cache, each decode call processes one token through the layers. Incremental detokenization uses $O(1)$ state and $O(\text{bytes})$ total work.

## In the wild

Production libraries split this same pipeline across a few helpers. `transformers` combines `apply_chat_template`, `generate`, and `TextIteratorStreamer`. `mlx-lm` pairs the tokenizer's template with `stream_generate` and `StreamingDetokenizer`. `llama.cpp` renders the same Jinja template in C++. All of them read stop-token IDs from model metadata rather than hardcoding them, because the end-of-turn marker varies by model (for Granite 4.0 it is `<|end_of_text|>`, ID 100,257).
