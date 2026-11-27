## The idea

Rendering a chat template is straightforward string assembly, so the main requirement is
getting every character and segment boundary right. Three principles explain why the
specification is written the way it is.

**The layout is a contract with the weights.** During fine-tuning, the model saw
`<|im_start|>`, a role, a newline, the content, `<|im_end|>`, and a trailing newline
thousands of times, and learned to emit `<|im_end|>` when its turn ends. Every character
in that frame is part of what the model conditions on. The generation prompt is the same
frame cut off right after `assistant\n`, so the first token the model generates is the
first token of its reply. Without the generation prompt, the model may continue the
user's turn or generate a role name itself. Adding it after a conversation that already
ends with an assistant turn causes the model to generate a second reply.

**Provenance decides what is special.** You could try to render the full string first
and scan it for special tokens, or you can record which pieces are special while
rendering. Only the second approach is safe, because scanning cannot distinguish a
template-written `<|im_end|>` from one typed by a user. Building the segment list first
tags every piece at the moment its origin is known, and the final string is just the
concatenation of the segments. Message content always goes through the `text` path and
is never inspected.

**Canonical segments preserve tokenization.** Merging adjacent text into one segment is
necessary for exact tokenization, not just cleanliness. A BPE tokenizer run on
`"user\nHi!"` can merge tokens across the newline, whereas running it on `"user\n"` and
`"Hi!"` separately prevents that merge. Segment boundaries must occur only at special
tokens so that segment-by-segment tokenization produces the exact same token ids as the
model's tokenizer on a trusted prompt.

## Reference solution (Python)

```python
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DEFAULT_SYSTEM = "You are a helpful assistant."


def render_chat(messages, add_generation_prompt, bos):
    segments = []

    def special(token):
        segments.append(("special", token))

    def text(piece):
        if not piece:
            return
        if segments and segments[-1][0] == "text":
            segments[-1] = ("text", segments[-1][1] + piece)
        else:
            segments.append(("text", piece))

    if bos is not None:
        special(bos)
    if not messages or messages[0]["role"] != "system":
        special(IM_START)
        text("system\n" + DEFAULT_SYSTEM)
        special(IM_END)
        text("\n")
    for i, message in enumerate(messages):
        role, content = message["role"], message["content"]
        if role == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                special(IM_START)
                text("user")
            text("\n<tool_response>\n")
            text(content)
            text("\n</tool_response>")
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
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
        text("assistant\n")
    return "".join(piece for _, piece in segments), segments
```

## Common mistakes

- **Missing a newline.** Writing `<|im_start|>assistant` without `\n` looks almost
  right, and the model often compensates by emitting the newline itself, wasting a
  generation step on a token it was never trained to predict there. The newline after
  `<|im_end|>` is equally required.
- **Always adding the generation prompt.** The flag exists because the same template
  also renders completed conversations for training, scoring, or prefix caching, where
  the final turn is already closed.
- **Emitting two BOS tokens.** Across two libraries, this happens when the template
  writes `bos` and the tokenizer's default `add_special_tokens=True` adds a second one.
  Inside one function, it happens when a helper that renders the first turn also writes
  `bos`. The model was only trained with a single BOS at position 0.
- **Scanning the rendered string for special tokens.** Splitting the final string on
  special-token substrings passes every benign test and fails the exact injection attack
  the segment list is meant to prevent.
- **Stripping whitespace from `content`.** Some chat templates (such as Llama 3) call
  `.strip()`, but ChatML templates do not. Stripping corrupts prompts that begin with
  indentation or end with a newline, such as code blocks.
- **Opening a separate turn for each tool message.** Parallel tool calls produce
  consecutive `"tool"` messages. This layout groups each maximal run into a single user
  turn with one `<tool_response>` block per message.
- **Rendering `"tool"` as the turn role.** Using `<|im_start|>tool` follows the generic
  rule used by some other ChatML variants, but this layout uses `<|im_start|>user` for
  tool responses.
- **Checking whether *any* message is `"system"` instead of only the first.** The
  default system turn depends strictly on `messages[0]`. A `"system"` message later in
  the list is rendered in place and does not suppress the default system turn.
- **Leaving unmerged or empty text segments.** Appending the role, content, and newline
  as separate `"text"` segments produces the right string but the wrong segmentation,
  and creates empty segments when `content` is `""`.

## Complexity

Time and space are linear in the total length of the message contents. Extending the
last text segment with `+` is quadratic in theory for a very long run of text pieces,
which can be avoided by buffering pieces in a list and joining once per segment, though
at these sizes the difference is negligible.

## In the wild

Production chat templates are Jinja programs stored in `tokenizer_config.json` (or a
separate `chat_template.jinja` file) and executed by `apply_chat_template` in
`transformers` or by built-in Jinja engines in servers such as `llama.cpp`. For plain
messages, the layout in this problem matches real released templates:

- Qwen2 and Hermes 3 use this exact three-role frame and insert
  `You are a helpful assistant.` whenever the first message is not a system message.
  Hermes 3 starts with `{{bos_token}}` (`<|begin_of_text|>`), while Qwen models have no
  BOS token.
- Qwen2.5 uses the same frame with its own default system message (`You are Qwen,
  created by Alibaba Cloud. You are a helpful assistant.`) and groups tool results into
  `<tool_response>` blocks inside a single user turn, matching this problem. Qwen3 keeps
  the same tool format and drops the default system message.
- SmolLM2 uses the same frame with its own default system message.

The reference solution here was tested against the Qwen2, Hermes 3, and Qwen2.5
templates on random conversations (swapping in each model's default system message) and
produces identical strings.

Because "ChatML" refers to a family of templates with subtle differences, a server must
always run the model's own template rather than a generic one. Special-token injection
is also a real issue in these stacks. `apply_chat_template` returns a plain string, so
tokenizing it parses every special-token substring, including those from user input.
Servers mitigate this by tokenizing user content separately from template markers, or by
escaping or rejecting special-token strings in user input.
