Suppose a piece ends with `<to`. You cannot emit it (it may be a tag) and you cannot
drop it (it may be text). What is the least you have to remember to do the right
thing when the next piece arrives, whichever way it goes? Now ask the same question
for a piece that ends with a backslash inside an argument string, and for one that
ends in the middle of the name.

---

Process the piece one character at a time through a small state machine with four
modes: content, the head of a call (up to the `{` that opens the arguments), the
arguments, and the tail (up to the end of `</tool_call>`). The state is an immutable
tuple: the mode, the next call index, a small buffer, and for JSON the nesting depth,
"inside a string", and "the previous character was a backslash". Because every flag
lives in the state, a boundary between pieces changes nothing: that is the whole
trick behind chunking invariance. Collect events in a list, appending to the previous
event when it is the same kind of delta.

---

Content mode: append the character to the buffer; if the buffer is `<tool_call>`,
switch to the head. Otherwise find the longest suffix of the buffer that
`"<tool_call>".startswith`, emit what is in front of it, and keep the suffix (do not
simply clear the buffer on a mismatch: in `<<tool_call>` the second `<` starts the
tag). Head mode: track strings with the escape flag; the second completed string
literal is the name (`json.loads` it and emit the start), and the first `{` outside
a string after the third literal opens the arguments at depth 1. Arguments mode:
emit every character; outside strings `{` and `[` add one to the depth, `}` and `]`
subtract one, and depth 0 ends the arguments. Tail mode: buffer characters until the
buffer ends with `</tool_call>`, emit the end, add one to the index. `parse_end`
returns the buffer as content if the mode is content and the buffer is not empty.
