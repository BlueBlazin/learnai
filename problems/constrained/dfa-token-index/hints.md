In `dfa-token-mask`, what did the answer depend on besides the automaton and the
vocabulary? If the only other input is the state, and there are finitely many states,
when can all the answers be computed? And if `b't'`, `b'tr'`, `b'tru'`, and `b'true'`
are all walked from the same state, how many of those steps are really different?

---

Two structures. First, a trie over the byte strings of the tokens (skip tokens with no
bytes): each node is one distinct prefix, its children are indexed by the next byte,
and a node remembers the token, if any, whose bytes end exactly there. An interior
node can hold a token too. Second, the index itself: for every state in `dfa.states`,
a dict from token to next state. Fill the second by walking the first.

---

For one state `q`, walk the trie and the automaton together, carrying the automaton
state that belongs to each trie node:

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

Each trie edge is stepped at most once per state, and never below a dead edge, which
is exactly the budget. The queries are then `sorted(index[state])` and a dict lookup
that turns a missing key into `ValueError`.
