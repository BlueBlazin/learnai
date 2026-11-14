Suppose you have read `[{"a": [1` and the next character is `}`. What do you need to
remember about the text so far to know that this is wrong, while `]` would be right?
How much of that memory do you need for a text that nests a thousand levels deep, and
which part of it do you ever look at?

---

Split the state in two. A **stack** of the open containers (`{` or `[`), of which only
the top matters at any moment. And a small **mode** that says what may come next:
a value, a value or `]` (right after `[`), a key or `}` (right after `{`), a key
(after a comma in an object), a colon, "after a value" (comma or closer, chosen by
the top of the stack), or the inside of a token: a string, an escape, the hex digits
of `\u`, a literal with some letters still to come, and the stages of a number
(after `-`, after a leading `0`, in the integer digits, after `.`, in the fraction,
after `e`, after the exponent's sign, in the exponent). `json_step` is one big
dispatch on the mode. Keep the stack immutable, a string or a tuple, and build a new
one on push and pop, so that old states stay valid.

---

Three details finish it.

A number ends without a closing character. If the mode is one in which the number is
already complete (`0`, integer digits, fraction digits, exponent digits) and the
character does not continue it, switch the mode to "after a value" and process the
same character again under that mode.

`json_complete` is: the stack is empty, and the mode is "after a value" or one of
those four complete-number modes. Nothing else, in particular not the start state.

After a comma go to "key" or "value", not back to the mode that follows the opening
bracket, since that one also accepts the closer and would let `[1,]` through.
