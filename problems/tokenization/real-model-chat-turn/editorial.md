## The idea

Three small pieces, in a fixed order.

**The template is data about the model.** Granite 4.0 ships a Jinja template of about
a hundred lines; for plain messages it reduces to three rules. Every prompt opens with
a system turn; a leading system message replaces the default text instead of adding a
second system turn (and an empty one does not count); and the prompt ends in an open
assistant turn with nothing after `<|end_of_role|>`. Rendering is then a join over
`(role, content)` pairs. Encoding the whole string once is safe here because the
contents are trusted. With untrusted content you would encode the template's special
tokens and the contents separately, which is the lesson of `chat-template-render`.

**The loop has two exits, and they differ by one call.** The model call is the
expensive part, so the loop is arranged around when a call is *needed*: logits are
needed only if another token will be chosen from them. After the end-of-turn token no
token follows, so it is never fed. After the `max_new_tokens`-th token no token
follows either, so that one is not fed. The number of calls is therefore one for the
prompt plus one per generated token, minus one if the limit ended the turn.

**Text is a byte stream.** A token is a byte string. The decoder's state is the
unfinished character at the end of the stream, at most three bytes; everything before
it is final and can be sent. An incremental UTF-8 decoder is exactly that state
machine. Special tokens have no bytes in the reply, and the flush at the end turns a
character that will never be finished into U+FFFD, so the concatenation of the pieces
always equals the decoding of the whole reply.

The yield comes before the next `step`. A generator makes this natural: the consumer
receives the piece, and the forward pass for the next token only starts when it asks
for more.

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

- **No generation prompt.** The prompt ends after the user's `<|end_of_text|>\n`, and
  the model is free to continue with anything, often another user turn.
- **A newline after the assistant header.** ChatML has one; Granite does not. One
  extra token, and every reply is conditioned on a prefix the model never saw.
- **No default system turn**, **the first system message rendered twice**, or **an
  empty system message kept as an empty turn.** All three are the same rule read
  carelessly: there is exactly one opening system turn, and it is never empty.
- **Stripping content.** The template inserts it verbatim.
- **The stop token as text.** `<|end_of_text|>` printed at the end of every reply is
  the most visible chat bug there is. The related one, **stopping on any special
  token**, cuts replies short when the model emits one in the middle.
- **Special tokens shown as text.** `tokenizer.token_bytes` returns the name of a
  special token, which is what a debugger wants and a user does not.
- **Decoding token by token.** `token_bytes(t).decode(errors="replace")` turns the
  emoji of the example into four `�`. On the real model this is every emoji and a good
  part of Japanese and Chinese text.
- **No flush.** A reply cut by the limit in the middle of a character loses it
  silently; the whole-reply decoding has a `�` there, so the stream and the stored
  transcript disagree.
- **Empty pieces.** A piece per token, empty or not, makes every client filter.
- **One token too many**, or **a call after the last token.** Both are off-by-one
  errors in the loop's exits; the second is invisible in the text and costs a forward
  pass per request.
- **The prompt fed token by token.** Forty forward passes instead of one. Prefill is
  a single call.
- **Buffering.** Collecting the pieces and yielding them at the end is a correct
  transcript and a broken stream: time to first token becomes the time of the whole
  reply.
- **Token ids in 16 bits.** A compact array of unsigned shorts holds the miniature
  tokenizer's ids and overflows on Granite's first prompt token, 100264. Vocabularies
  passed 65,536 long ago.

## Complexity

One model call over the $P$ prompt tokens and at most `max_new_tokens` single-token
calls; with the model's KV cache, each of those costs one token through the layers.
Detokenization is $O(1)$ state and $O(\text{bytes})$ work in total.

## In the wild

`transformers` splits this into `apply_chat_template`, `generate`, and a
`TextIteratorStreamer`; `mlx-lm` into the tokenizer's template and
`stream_generate` with its `StreamingDetokenizer`; `llama.cpp` renders the same Jinja
template in C++. All of them stop on a set of end-of-turn ids read from the model's
files, because the stop token is model data too: for Granite 4.0 it is
`<|end_of_text|>`, id 100257, and for the next model it is something else.
