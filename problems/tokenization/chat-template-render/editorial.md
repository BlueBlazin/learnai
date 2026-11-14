## The idea

Rendering a template is string concatenation, and the difficulty is exactness, not
algorithm. Three things are worth understanding rather than memorizing.

**The layout is a contract with the weights.** During fine-tuning the model saw
`<|im_start|>`, a role, a newline, the content, `<|im_end|>`, a newline, thousands of
times, and learned to produce `<|im_end|>` when its own turn is complete. Every
character of that frame is part of what the model conditions on. The generation
prompt is the same frame cut off after the newline that follows `assistant`: the next
token the model produces is the first token of its answer. Without the generation
prompt the model may continue the user's turn or write a role name itself; with it
appended to a conversation that already ends in a complete assistant turn, the model
answers a second time.

**Provenance decides what is special.** There are two ways to find the special tokens
of a prompt. One is to render a string and search it. The other is to remember, while
rendering, which pieces the template wrote as special tokens. Only the second is
correct, because the first cannot tell the template's `<|im_end|>` from one that a
user typed. So the renderer builds the segment list first, tagging each piece at the
moment it is written, when its origin is known, and the string is derived from the
list. Content goes through the `text` path whatever it contains, and no code ever
inspects it.

**Canonical segments.** Merging adjacent text into one segment is more than tidiness:
a tokenizer applied to `"user\nHi!"` may merge across the newline, and applied to
`"user\n"` and `"Hi!"` separately it cannot. The segment boundaries must be the
special tokens and nothing else, so that tokenizing segment by segment gives the same
ids as the model's own tokenizer gives for a trusted prompt.

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

- **A missing newline.** `<|im_start|>assistant` without the `\n` looks complete, and
  the model usually copes by emitting the newline itself, at the price of a first
  token it was never asked to predict in training. The newline after `<|im_end|>` is
  just as much part of the format.
- **Always adding the generation prompt.** The flag exists because the same template
  renders finished conversations, for training, scoring, or caching a prefix, where
  the last turn is already there.
- **Two BOS tokens.** The classic version involves two libraries: the template writes
  `bos`, and the tokenizer, called with its default of adding special tokens, adds
  another. Inside one function it happens when a helper that renders "the first turn"
  also writes BOS. The model has only ever seen one BOS at position 0.
- **Parsing content for specials.** Deriving the segments by splitting the finished
  string on special-token strings gives the right answer for every honest
  conversation and the wrong one for the attack it is supposed to stop.
- **Trimming content.** Some templates do strip whitespace (Llama 3's does); ChatML
  templates do not. Applying `.strip()` out of habit changes prompts that end in a
  newline or begin with indentation, such as code.
- **One turn per tool message.** Parallel tool calls produce several results in a
  row; the format puts them in one user turn, each in its own `<tool_response>` block.
- **Rendering `tool` as a role.** `<|im_start|>tool` is what the generic rule gives,
  and it is what some other ChatML models expect, but not this layout.
- **"Is there a system message?" instead of "is the first message one?"** The default
  system turn depends only on the first message. A system message in the middle of a
  conversation is rendered where it stands and does not replace the default.
- **Unmerged or empty text segments.** Appending role, content, and newline as three
  segments gives the right string and the wrong segmentation, including an empty
  segment for empty content.

## Complexity

Linear in the total length of the contents. Appending to the last text segment by
string concatenation is quadratic in principle for a very long run of text pieces;
collecting pieces in a list and joining once per segment avoids that, and does not
matter at these sizes.

## In the wild

Real templates are Jinja programs stored in `tokenizer_config.json` (newer releases
ship a separate `chat_template.jinja`) and run by `apply_chat_template` in
`transformers` and by the Jinja engines built into `llama.cpp` and other servers. The
layout here is what those programs do for plain messages:

- Qwen2 and Hermes 3 use exactly the three-role frame and insert
  `You are a helpful assistant.` when the first message is not a system message.
  Hermes 3's template begins with `{{bos_token}}`, which for that model is
  `<|begin_of_text|>`; Qwen models have no BOS token at all.
- Qwen2.5 uses the same frame with a different default system message (`You are Qwen,
  created by Alibaba Cloud. You are a helpful assistant.`) and renders tool results as
  the grouped `<tool_response>` blocks in a user turn that this problem specifies.
  Qwen3 keeps the tool format and drops the default system message.
- SmolLM2 uses the frame with its own default system message.

The reference of this problem was compared with the Qwen2, Hermes 3, and Qwen2.5
templates on random conversations (with the default system message set to each
model's own) and produces identical strings.

So "ChatML" names a family, and a server must run the model's own template, not a
generic one. The injection problem is real in these stacks: `apply_chat_template`
returns a string, and tokenizing it parses every special-token string in it, including
those from user content. Servers defend by tokenizing content separately from the
template's markers, or by escaping or rejecting special-token strings in user input.
