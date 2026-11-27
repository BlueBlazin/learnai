Suppose you have read `[{"a": [1` and the next character is `}`. What do you need to remember about the text so far to know that `}` is invalid while `]` is valid? How much memory does that take for a text nested a thousand levels deep, and which part of it do you inspect at each step?

---

Split the state into two parts:

- A **stack** of open containers (`{` or `[`), where only the top entry matters at any step.
- A small **mode** describing what may come next. Structural modes cover expecting a value, a value or `]` (right after `[`), a key or `}` (right after `{`), a key (after a comma in an object), a colon, or "after a value" (where the top of the stack picks the comma target and closer). Token-interior modes cover strings, escapes, `\u` hex digits, remaining literal characters, and the stages of a number (after `-`, after a leading `0`, integer digits, after `.`, fraction digits, after `e`/`E`, after an exponent sign, and exponent digits).

`json_step` dispatches on the current mode. Store the stack in an immutable value such as a string or tuple, creating a new stack on push and pop so earlier states remain unchanged.

---

Three final details tie everything together:

1. **Number termination.** A number has no closing delimiter. When the mode is one where the number is already complete (`0`, integer digits, fraction digits, or exponent digits) and the next character cannot continue the number, switch the mode to "after a value" and process that same character under the new mode.
2. **Completion.** `json_complete` is `True` if and only if the stack is empty and the mode is either "after a value" or one of the four complete-number modes. The initial start state must not be complete.
3. **Commas.** After a comma, transition to "key" or "value" rather than back to the post-opening-bracket mode, because the opening-bracket mode also accepts a closing bracket and would allow trailing commas like `[1,]`.
