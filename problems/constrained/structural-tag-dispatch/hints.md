Ignore tokens at first and think about reading the output one byte at a time. How many distinct phases can the reader be in, and what is the minimum information it must remember in each phase to judge the next byte? In free text, which part of the bytes read so far can still participate in a future trigger match?

---

Four state kinds are enough, each represented as a small tuple:

- **Free text:** the longest suffix of the current free text that is a proper prefix of some trigger.
- **Begin:** the tuple of still-matching tag candidates and how many bytes of their `begin` string have been matched.
- **Content:** the active tag and the current state of its `inner` automaton.
- **End:** the active tag and how many bytes of its `end` string have been matched.

Write a single byte-level transition function `step(state, byte)` that returns the next state or `None`. A non-empty token is allowed if stepping through all of its bytes never returns `None`. `advance` returns the state after the last byte, and `is_accepting` is true if and only if the state is in free text.

---

The four byte transitions work as follows:

- **Free:** Append `byte` to the held suffix. While the result is not a prefix of any trigger, drop its first byte (do not clear it completely, because with trigger `<<fn=`, reading `<<<` must keep `<<`). If the resulting suffix equals a trigger, collect the tags whose `begin` starts with it and enter the begin phase with `matched = len(trigger)`.
- **Begin:** Keep only the candidates whose `begin[matched] == byte`. If none remain, return `None`. If the remaining candidate's `begin` is now complete (`matched + 1 == len(begin)`), transition to the content phase at `inner.start`.
- **Content:** From the current `inner` state `q`, compute `after = inner.step(q, byte)`. If `after != DEAD`, stay in content at `after`. Otherwise, if `inner.is_accepting(q)` and `byte == end[0]`, transition to the end phase with 1 byte matched. Otherwise, return `None`.
- **End:** Require `byte == end[matched]`. Once the last byte of `end` is matched, return to the free state with an empty suffix `b""`.

Check for completion immediately whenever the matched count is updated (both in begin and in end). That way, a trigger equal to an entire `begin` string and a one-byte `end` string need no special cases. In `allowed`, skip tokens whose byte string is empty.
