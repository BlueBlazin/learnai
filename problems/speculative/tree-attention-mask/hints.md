Ignore the rest of the tree for a moment and focus on a single node. If its root path were the only continuation of the prefix, the model would run `prefix + path` as an ordinary causal sequence. Think about which tokens that node would attend to and what position ID it would have. The flattened tree must give every node that exact same view, regardless of where it sits in the flat order.

---

For every node, you need both its set of ancestors and its depth. You can compute both at once by climbing parent pointers across all nodes in parallel. Start with a vector `anc = arange(N)` so every node starts on itself. On each step, mark `reach[i, anc[i]] = True` for every node still in the tree (`anc[i] >= 0`), add 1 to its step count, and move up with `anc = parents[anc]` while keeping `-1` entries fixed at `-1`. The loop finishes in $(\text{max depth} + 1)$ steps. At the end, `reach` is the tree-to-tree mask block (including the diagonal), and each node's depth is its step count minus 1.

---

Assemble `mask` from four blocks: prefix-to-prefix is the standard lower triangle, prefix-to-tree is all `False`, tree-to-prefix is all `True`, and tree-to-tree is `reach`. Set `position_ids` to `arange(P)` concatenated with `P + depth`.

For `verify_tree`, start with `current = -1` and `want = target_argmax[0]`. While some node satisfies `parents == current` and `tokens == want`, append that node to `path`, move `current` to that node, and update `want = target_argmax[node + 1]`. Return `path` and the final `want` (remembering that entry `0` belongs to the prefix, so node `i` maps to `i + 1`).
