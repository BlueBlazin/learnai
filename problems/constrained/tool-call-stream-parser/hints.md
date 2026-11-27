Suppose a piece ends with `<to`. You cannot emit it because it might start a tag, and you cannot drop it because it might be ordinary text. What is the minimum information you must store in the state to handle the next piece correctly in either case? Ask the same question for a piece ending with a backslash inside an argument string, and for a piece ending in the middle of the tool name.

---

Process `piece` one character at a time through a small state machine with four modes: `content`, `head` (up to the `{` that opens `ARGS`), `arguments`, and `tail` (up to the end of `</tool_call>`). Store the state in an immutable tuple containing the mode, the next call index, a small buffer, and three JSON tracking fields: nesting depth, `in_string`, and `escaped`. Because every flag lives in the state, boundaries between pieces cannot change the result, which guarantees chunking invariance. Collect events in a list and merge consecutive deltas of the same kind and call index.

---

In `content` mode, append each character to the buffer. If the buffer equals `<tool_call>`, switch to `head` mode. Otherwise, find the longest suffix of the buffer that `"<tool_call>"` starts with, emit everything before that suffix, and keep the suffix. Do not clear the buffer blindly on a mismatch, because in `<<tool_call>` the second `<` begins the tag.

In `head` mode, track JSON strings using `in_string` and `escaped`. The second completed string literal is `NAME` (decode it with `json.loads` and emit `tool_call_start`), and the first `{` outside a string after the third literal opens `arguments` mode at depth 1.

In `arguments` mode, emit every character. Outside strings, `{` and `[` increment depth while `}` and `]` decrement it, and depth 0 switches to `tail` mode.

In `tail` mode, buffer characters until the buffer ends with `</tool_call>`, then emit `tool_call_end`, increment `index`, and return to `content` mode. In `parse_end`, return the buffer as a `content` event if the mode is `content` and the buffer is non-empty.
