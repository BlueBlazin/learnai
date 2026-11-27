In `dfa-token-mask`, besides the automaton and vocabulary, what else did the allowed tokens depend on? Since there are finitely many states, when can you compute all of those answers? Also, when `b't'`, `b'tr'`, `b'tru'`, and `b'true'` are all walked from the same state, how many of their automaton steps are actually distinct?

---

Use two data structures. First, build a prefix tree (trie) over the non-empty byte strings in `vocab`. Each node represents a distinct prefix, maps each next byte to a child node, and stores any token ids whose byte strings end at that node (including interior nodes). Second, build the index itself as a dictionary mapping every state in `dfa.states` to a dictionary of `{token: next_state}` pairs, filled by walking the trie from each state.

---

For each state `q`, walk the trie and the automaton together while carrying the current automaton state for each trie node:

```python
stack = [(root, q)]
while stack:
    node, at = stack.pop()
    for byte, child in node.children.items():
        after = dfa.step(at, byte)       # one step per trie edge, for this q
        if after == DEAD:
            continue                     # the whole subtree dies: do not descend
        ...record child's token -> after, then push (child, after)
```

Each trie edge is stepped at most once per state and never below a `DEAD` edge, which matches the step budget exactly. `allowed_from_index` then returns `sorted(index[state])`, and `advance_from_index` looks up `index[state][token]` and raises `ValueError` on a missing key.
