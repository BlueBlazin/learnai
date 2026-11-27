Many choices begin with the same tokens. Which data structure stores shared prefixes once and lets you find all valid next tokens by walking down from the root?

---

Represent each node as a dict mapping a token ID to its child node. To insert a choice, start at the root and follow one edge per token, creating missing nodes along the way. To answer a query, walk from the root following one edge per generated token. If any edge is missing, return `[]`. Otherwise, the keys of the node you reach are the tokens that can extend the prefix. You still need to track where complete choices end, keeping in mind that a node where a choice ends may also have children.

---

Record the end of a choice directly in its node, independently of its children, by adding an edge labeled `eos_id` (since no choice contains `eos_id`, it cannot collide with a real token).

```python
node = root
for token in choice:
    node = node.setdefault(token, {})
node.setdefault(eos_id, {})
```

With that edge in place, `sorted(node)` returns all valid next tokens plus `eos_id` when a choice ends there, deduplicated and in increasing order.
