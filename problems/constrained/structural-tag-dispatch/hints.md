Forget tokens for a moment and read the output one byte at a time. How many
different situations can the reader be in, and what is the least it must remember
in each to judge the next byte? In free text in particular: which part of what has
been read can still matter for a trigger?

---

Four kinds of state are enough, each a small tuple: free text (remember the longest
suffix of it that is a proper prefix of some trigger), begin (which tags are still
candidates and how many bytes of their `begin` have been matched), content (which tag,
and the state of its automaton), and end (which tag, and how many bytes of `end` have
been matched). Write one function `step(state, byte)` that returns the next state or
`None`. Then a token is allowed if stepping through all of its bytes never yields
`None`, `advance` returns the state after the last byte, and the output is complete
exactly in a free-text state.

---

The four transitions:

- Free: append the byte to the held suffix; while the result is not a prefix of any
  trigger, drop its first byte (do not clear it: with the trigger `<<fn=`, the text
  `<<<` must keep `<<`). If what is left equals a trigger, the candidates are the tags
  whose `begin` starts with it and the matched length is the trigger's length.
- Begin: keep the candidates whose `begin` has this byte at the matched position; none
  left is `None`. If a candidate's `begin` is now complete, go to its content with
  `inner.start`.
- Content: `inner.step(q, byte)`; if it is not `DEAD`, stay. Otherwise, if
  `inner.is_accepting(q)` and the byte is `end[0]`, go to the end state with one byte
  matched; else `None`.
- End: the byte must equal `end[matched]`. After the last one return the free state
  with an empty suffix.

Handle "the begin is already complete" and "the end is already complete" right where
the count reaches the length, so that a trigger equal to a whole `begin` and a
one-byte `end` need no special cases. Skip tokens with empty bytes in `allowed`.
