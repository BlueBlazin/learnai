Run one turn of a chat with a real model, IBM's Granite 4.0 350M: render the
conversation in the model's own prompt layout, let the model read it, generate
greedily until the model ends its turn, and stream the reply as text. The tokenizer
and the model call are given. What is left is everything a chat endpoint does around
them, where each mistake still produces an answer, only a worse one.

## Specification

```python
def chat_turn(messages, tokenizer, step, max_new_tokens): ...  # a generator of str
```

- `messages`: the conversation, a non-empty list of dicts with the keys `"role"`
  (`"system"`, `"user"`, or `"assistant"`) and `"content"` (a string).
- `tokenizer`: the model's tokenizer, a
  [`ByteLevelBPE`](../../../python/src/whetstone/kit/bpe.py) from the kit.
  `tokenizer.encode(text)` returns a list of ints, and an added token written in the
  text (such as `<|end_of_text|>`) becomes its single id; `tokenizer.token_bytes(i)`
  returns the bytes of one token; `tokenizer.specials` is a dict from the text of each
  special token to its id.
- `step`: the model. `step(token_ids)` takes a non-empty list of Python ints, feeds
  those tokens to the model after everything fed before, and returns the logits of
  the token that comes next: a float32 numpy array of shape `(V,)`. The model keeps
  its own state (a KV cache), so nothing is ever fed twice.
- `max_new_tokens`: a Python int, at least 1.
- `chat_turn` is a generator. It yields the reply as non-empty strings.

**1. The prompt.** Granite's layout marks each turn with three special tokens:

```
<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n
```

where `\n` is one newline character. The prompt text is the concatenation of:

1. **A system turn, always.** If the first message has the role `system` and its
   content is not empty, that content is the system text, and the message is not
   rendered again in part 2. Otherwise the system text is the default,

   ```
   You are a helpful assistant. Please ensure responses are professional, accurate, and safe.
   ```

   (a first system message with empty content is dropped in favor of it).
2. **One turn per remaining message**, in order, with its own role. A `system`
   message that is not the first message is an ordinary turn with the role `system`.
3. **The generation prompt**: `<|start_of_role|>assistant<|end_of_role|>`, with
   nothing after it. The turn is left open for the model.

Content is inserted verbatim: no stripping, also when it is empty. The prompt ids are
`tokenizer.encode(text)` of the whole text; nothing is added in front.

**2. Generation.** Call `step` once with all prompt ids. Then repeat: the next token
is the index of the largest logit. If it is the id of `<|end_of_text|>`
(`tokenizer.specials["<|end_of_text|>"]`), the turn is over: the token is not text,
and it is not fed to `step`. Otherwise it is a generated token; if it is the
`max_new_tokens`-th, the turn is over as well, and otherwise call `step` with a list
holding just that token. So `step` is called once for the prompt and once for every
generated token except, when the limit ends the turn, the last one: a call whose
logits nobody reads is a forward pass wasted.

**3. Text.** Tokens are byte strings, and one character's bytes may be spread over
several tokens. Let the bytes of the reply be the bytes of its generated tokens in
order, where a token in `tokenizer.specials` contributes no bytes (it is fed to
`step` like any other, and counts toward the limit). After each generated token,
yield the text that has become final: everything decoded as UTF-8, invalid bytes as
U+FFFD, **except an unfinished character at the very end**, which waits for its
remaining bytes. When the turn is over, an unfinished character that is still
waiting is yielded as one U+FFFD. Never yield an empty string: a token that only
starts or continues a character yields nothing.

This is the rule of `streaming-detokenizer`, and Python's incremental decoder
implements it: `codecs.getincrementaldecoder("utf-8")(errors="replace")`, with
`decode(data)` per token and `decode(b"", final=True)` at the end.

**4. Streaming.** Yield the text of a token *before* calling `step` with that token.
A consumer that stops reading must not have paid for a forward pass it never saw the
result of, and one that keeps reading sees each piece a model call earlier.

Write the template out by hand: a template engine (`jinja2`) and the `transformers`
and `tokenizers` packages are not allowed.

## Example

`messages = [{"role": "user", "content": "Hi!"}]`, `max_new_tokens = 16`. There is no
system message, so the prompt text is (one line break shown per `\n`; the text ends
right after `<|end_of_role|>`):

```
<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Hi!<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
```

The first public case uses a miniature tokenizer of 382 tokens, under which this text
is 54 ids, and a scripted model that emits the ten tokens below.

| call of `step` | argmax of the logits it returns (bytes) | yielded |
|---|---|---|
| the 54 prompt ids | `48` | `"H"` |
| that token | `65 6C` | `"el"` |
| that token | `6C 6F` | `"lo"` |
| that token | `21` | `"!"` |
| that token | `20` | `" "` |
| that token | `F0` | nothing |
| that token | `9F` | nothing |
| that token | `99` | nothing |
| that token | `82` | `"🙂"` |
| that token | `<|end_of_text|>` | (the generator ends) |

Ten calls, six pieces, and the reply is `"Hello! 🙂"`. The emoji is the four bytes
`F0 9F 99 82`, one token each in this tokenizer; decoding the tokens one by one would
yield four replacement characters. With `max_new_tokens = 7` the turn would end after
the token `9F`: `step` is called seven times (the seventh generated token is not
fed), and the pieces are `"H"`, `"el"`, `"lo"`, `"!"`, `" "`, `"�"`.

On the real model, the conversation "Reply with three different emoji and nothing
else." is a prompt of 39 tokens, and the reply is nine tokens, three per emoji
(`F0 9F`, `91`, `8D` for the first): the pieces are `"👍"`, `"👎"`, `"👎"`.

## Constraints

- Everything is compared exactly: the token ids of every call of `step`, and the
  list of yielded strings. The expected prompts are the model's own Jinja template
  rendered by `transformers`; the expected real replies are its greedy `generate`.
- At most 8 messages; contents contain no special-token text, and may be empty or
  contain spaces and newlines. The logits always have a unique maximum.
- The cases that need no download use the miniature tokenizer (the same pre-tokenizer
  and special-token names as Granite's, with other ids) and scripted logits; token ids
  must be looked up, never written as literals. Scripted replies contain invalid
  UTF-8 and special tokens other than `<|end_of_text|>`; they never put a byte
  `A0..BF` after `ED`, the one corner in which decoders disagree.
- The real tokenizer has 100,352 tokens, and its special tokens have ids above
  100,255. Real prompts have 33 to 49 tokens; reading one takes the kit's numpy model
  about 0.3 seconds. The time limit per case is 20 seconds.

## Why this matters

A chat model is a text-completion model that was fine-tuned on one exact byte layout,
and the chat endpoint is the code that reproduces it. Granite's layout is not ChatML:
other markers, a system turn that is always present, no newline after the assistant
header. Send ChatML to it, or forget the generation prompt, and it still answers,
in a voice it was never trained to use; evaluations quietly drop a few points and
nobody finds out why. The streaming half fails just as quietly: decode token by token
and every emoji and many CJK characters arrive as `�`; feed the end-of-turn token back
and the model starts talking to itself; buffer the reply and time to first token
becomes time to last token. This loop, done right, is what `tokenizer.apply_chat_template`
plus `generate` with a `TextStreamer` do in `transformers`, and what `mlx_lm.stream_generate`
does on a Mac.
