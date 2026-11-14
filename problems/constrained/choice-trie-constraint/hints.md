Many choices begin with the same tokens. Which structure stores sequences so that a
shared beginning is stored once, and "what can follow this prefix?" is answered by
walking down from the root?

---

Make each node a dict from token id to child node. To insert a choice, start at the
root and follow (creating when missing) one edge per token. To answer a query, start
at the root and follow one edge per generated token; if an edge is missing, the output
has left the trie and the answer is `[]`. Otherwise the keys of the node you reached
are the tokens that may follow. What remains is to record *where choices end*, and a
node where a choice ends may still have children.

---

Record the end of a choice in the node itself, separately from its children, for
example as one more edge labeled `eos_id` (no choice contains EOS, so it cannot
collide):

```python
node = root
for token in choice:
    node = node.setdefault(token, {})
node.setdefault(eos_id, {})
```

Then the answer to a query is `sorted(node)`: the children and, where a choice ends,
EOS, each once and in increasing order.
