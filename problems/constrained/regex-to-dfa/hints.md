Do not try to build the deterministic automaton directly from the pattern. Which intermediate representation lets you translate each construct (`|`, concatenation, `*`, `{m,n}`) independently with a small rule? And what does a state of the final DFA represent in terms of that intermediate object?

---

Break the compiler into four small stages:

1. **Parse** with recursive descent using one function per precedence level (`alternation` calls `concatenation`, which calls `repetition`, which calls `atom`). Reduce every basic atom to "one byte from this set", where a literal is a single-byte set and a class is a larger set. Expand "one character except ..." into an alternation of four byte-set sequences matching the UTF-8 table in the statement. Represent `{m,n}` as `m` required copies followed by `n - m` optional copies, and `{m,}` as `m` required copies followed by a loop.
2. **Thompson's construction.** Convert each AST node into an NFA fragment with one entry state and one exit state, wired together using epsilon transitions that consume no bytes.
3. **Subset construction.** Represent each DFA state as a set of NFA states closed under epsilon transitions. Start from the epsilon closure of the NFA entry. For each byte with outgoing edges, collect the target NFA states and take their epsilon closure. Explore new subsets breadth-first and assign each a state ID. A subset is accepting if it contains the NFA exit. Omit entries for bytes with no targets.
4. **Minimize**, using partition refinement below.

---

Minimize the DFA with Moore's partition refinement. Start with two blocks: accepting states and non-accepting states. Repeatedly assign each state the signature `(current_block, sorted list of (byte, target_block))` and group states with identical signatures into new blocks. Missing bytes simply do not appear in the list, which distinguishes rejection from any real transition. Stop when the number of blocks stops increasing, and emit one DFA state per block.

You do not need a separate dead-state trimming pass if the DFA comes from stages 1–3. Every NFA fragment state lies on a path to the exit, so every non-empty subset can still reach an accepting state, and breadth-first search only visits reachable subsets.

For the `n - m` optional copies in `{m,n}`, add an epsilon edge from the start of each optional copy directly to the fragment exit. Verify your copy count on a small pattern like `a{2,4}`, which should have five minimal states (0, 1, 2, 3, or 4 `a`s read, with the last three accepting).
