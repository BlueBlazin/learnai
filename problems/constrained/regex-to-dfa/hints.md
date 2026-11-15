Do not try to build the deterministic automaton directly from the pattern. Which
intermediate object lets you translate each construct (`|`, concatenation, `*`,
`{m,n}`) by itself, with a small rule per construct? And what is a state of the final
automaton in terms of that object?

---

Four stages, each small:

1. **Parse** by recursive descent, one function per precedence level:
   `alternation` calls `concatenation` calls `repetition` calls `atom`. Reduce every
   atom to "one byte from this set": a literal is a set of one byte, a class a
   larger set. "One character except ..." is an alternation of four sequences of
   byte sets, taken from the table in the statement. `{m,n}` is `m` copies in a row
   followed by `n - m` copies that may each be skipped; `{m,}` is `m` copies and a
   loop.
2. **Thompson's construction**: every tree node becomes a fragment with one entry
   and one exit state, glued together with transitions that read nothing.
3. **Subset construction**: a deterministic state is the set of fragment states
   reachable after some input, closed under the read-nothing transitions. Start from
   the closure of the entry; for each byte, collect the targets and close again.
   Explore breadth-first, numbering each new set. A set is accepting if it contains
   the exit. Bytes with no target get no entry.
4. **Minimize**, below.

---

Minimization by partition refinement (Moore): start with two blocks, accepting and
not accepting. Repeatedly give each state the signature
`(its block, sorted list of (byte, block of the target))` and regroup states by
signature; a missing byte is simply absent from the list, which is how "rejects"
is told apart from every real target. Stop when the number of blocks no longer
grows, and emit one state per block. No trimming pass is needed if the automaton
came from stages 1 to 3: every fragment state lies on a path to the exit, so every
set can still accept, and breadth-first exploration only creates reachable sets.

For the repetition fragment with `high - low` optional copies, give every optional
copy an edge that skips to the fragment's exit, and count the copies on a small
case such as `a{2,4}` (five states: 0, 1, 2, 3, 4 `a`s read, the last three
accepting).
