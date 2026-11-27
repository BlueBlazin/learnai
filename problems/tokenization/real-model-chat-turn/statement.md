Implement one turn of a chat with IBM's Granite 4.0 350M. Format the conversation into the model's prompt layout, run greedy decoding until the model finishes or hits the token limit, and stream the reply as UTF-8 text.

![This problem is the whole path, one chat turn from the messages to streamed text.](assets/map.svg)

## Specification

```python
def chat_turn(messages, tokenizer, step, max_new_tokens): ...  # a generator of str
```

- `messages`: a non-empty list of dicts, each with keys `"role"` (`"system"`, `"user"`, or `"assistant"`) and `"content"` (a string).
- `tokenizer`: the model's tokenizer, a [`ByteLevelBPE`](../../../python/src/learnai/kit/bpe.py) from the kit. `tokenizer.encode(text)` returns a list of Python ints and encodes any added special token written in `text` (such as `<|end_of_text|>`) as its single ID. `tokenizer.token_bytes(token_id)` returns the raw bytes of one token. `tokenizer.specials` is a dict mapping each special token's text to its ID.
- `step`: a callable representing the model. Calling `step(token_ids)` with a non-empty list of Python ints feeds those tokens after everything previously fed and returns the next-token logits as a `float32` NumPy array of shape `(V,)`, where `V` is the vocabulary size. The model keeps its own KV cache, so never feed the same token twice.
- `max_new_tokens`: a Python int, at least `1`.
- `chat_turn` is a generator that yields the reply as non-empty strings.

**1. Prompt formatting.** Granite marks each turn with three special tokens:

```
<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n
```

Here `\n` is a single newline character. Concatenate three parts to build the prompt text:

1. **A leading system turn (always present).** If the first message has role `"system"` and non-empty content, use that content as the system text and do not render the first message again in part 2. Otherwise, use the default system text:

   ```
   You are a helpful assistant. Please ensure responses are professional, accurate, and safe.
   ```

   If the first message has role `"system"` and empty content, drop it and use this default instead.
2. **One turn per remaining message**, in order, using each message's own role and content. A `"system"` message after the first message is rendered normally with role `"system"`.
3. **The generation prompt**, `<|start_of_role|>assistant<|end_of_role|>`, with nothing after it. This leaves the assistant turn open for the model to complete.

Insert every `"content"` string verbatim without stripping whitespace, even when it is empty. Produce the prompt IDs by calling `tokenizer.encode(text)` once on the full prompt string, without adding any extra token in front.

**2. Generation.** Call `step` once with all prompt IDs to obtain the initial logits. Then repeat:

- Choose the next token as the index of the largest logit.
- If that token is `tokenizer.specials["<|end_of_text|>"]`, stop the turn. Do not emit text for it and do not feed it to `step`.
- Otherwise, treat it as a generated token. If it is the `max_new_tokens`-th generated token, end the turn without calling `step` again. Otherwise, call `step([token])` to get the next logits.

In total, `step` is called once for the prompt and once for each generated token except the last one when `max_new_tokens` ends the turn. Calling `step` on that final token would run a forward pass whose logits are never read.

**3. Text decoding.** Tokens are byte strings, and a single UTF-8 character can span multiple tokens. Form the reply's byte stream from the generated tokens in order, where any token ID in `tokenizer.specials.values()` contributes zero bytes (`b""`). Special tokens other than `<|end_of_text|>` are still fed to `step` and still count toward `max_new_tokens`.

After each generated token, yield all newly finalized UTF-8 text, replacing invalid bytes with U+FFFD. Hold back any incomplete character at the end of the stream until later tokens supply its remaining bytes. When the turn ends, flush the decoder so any still-incomplete character is yielded as a single U+FFFD. Never yield an empty string.

This is the same rule as `streaming-detokenizer`, and Python's standard incremental decoder implements it via `codecs.getincrementaldecoder("utf-8")(errors="replace")`. Call `decode(data)` for each token and `decode(b"", final=True)` at the end.

**4. Streaming order.** Yield the text produced by a token *before* calling `step` with that token. A consumer that stops early avoids paying for an extra forward pass, and one that keeps reading receives each piece one model call sooner.

Write the prompt template by hand. Importing `jinja2`, `transformers`, or `tokenizers` is not allowed.

## Example

Take `messages = [{"role": "user", "content": "Hi!"}]` and `max_new_tokens = 16`. Because there is no system message, the prompt uses the default system text and ends right after `<|end_of_role|>` (one line break per `\n`):

```
<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>
<|start_of_role|>user<|end_of_role|>Hi!<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
```

The first public test uses a miniature tokenizer of 382 tokens and a scripted model. Under this tokenizer, the prompt encodes to 54 IDs, and the scripted model emits the ten tokens below.

| Call of `step` | Argmax of returned logits (bytes) | Yielded |
|---|---|---|
| the 54 prompt IDs | `48` | `"H"` |
| that token | `65 6C` | `"el"` |
| that token | `6C 6F` | `"lo"` |
| that token | `21` | `"!"` |
| that token | `20` | `" "` |
| that token | `F0` | nothing |
| that token | `9F` | nothing |
| that token | `99` | nothing |
| that token | `82` | `"🙂"` |
| that token | `<|end_of_text|>` | (generator ends) |

This takes ten calls to `step`, yields six pieces, and produces `"Hello! 🙂"`. The emoji is the four-byte UTF-8 sequence `F0 9F 99 82`, split across four tokens in this tokenizer. Decoding each token independently would have produced four replacement characters instead.

With `max_new_tokens = 7`, the turn ends right after the token `9F`. `step` is called seven times (the seventh generated token is not fed), and flushing the incomplete emoji at the end produces the pieces `"H"`, `"el"`, `"lo"`, `"!"`, `" "`, `"�"`.

On the real Granite model, the user message `"Reply with three different emoji and nothing else."` produces a 39-token prompt and a nine-token reply. Each emoji spans three tokens (for example, `F0 9F`, `91`, `8D` for the first), and the yielded pieces are `"👍"`, `"👎"`, `"👎"`.

## Constraints

- Both the token IDs passed to every call of `step` and the list of yielded strings are checked for exact equality. Expected prompts match Granite's Jinja template rendered by `transformers`, and expected real-model replies match its greedy `generate`.
- `messages` contains at most 8 messages. Message contents contain no special-token text and may be empty or include spaces and newlines. Logits always have a unique maximum.
- Tests that require no model download use a miniature tokenizer (same pre-tokenizer and special-token names as Granite, different IDs) and scripted logits. Always look up token IDs from `tokenizer` instead of hardcoding integers. Scripted replies include invalid UTF-8 and special tokens other than `<|end_of_text|>`, but never place a byte in `0xA0..0xBF` immediately after `0xED` (the one edge case where UTF-8 decoders disagree).
- The real Granite tokenizer has 100,352 tokens, with special-token IDs above 100,255. Real prompts have 33 to 49 tokens, and prefilling one takes about 0.3 seconds on the kit's NumPy model. The time limit is 20 seconds per case.

## Why this matters

A chat model is fine-tuned on one exact prompt byte layout, and the chat endpoint must reproduce it faithfully. Granite's layout differs from ChatML in its role markers, its mandatory system turn, and the absence of a newline after the assistant header. Sending ChatML or omitting the generation prompt still produces fluent output, but evaluation scores quietly drop because the model is running off-distribution.

The streaming layer fails just as quietly. Decoding token by token turns emojis and many CJK characters into `�`. Feeding the stop token back into the model can make it continue generating past the end of its turn. Buffering the entire reply before yielding turns time to first token into time to last token. Together, this prompt formatter, decode loop, and incremental detokenizer implement the core behavior of `tokenizer.apply_chat_template` plus `generate` with `TextStreamer` in `transformers`, and `mlx_lm.stream_generate` on macOS.
