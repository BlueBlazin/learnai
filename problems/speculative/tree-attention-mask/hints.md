Forget the tree for a moment and think about one node. If its root path were the
only continuation of the prefix, the model would process `prefix + path` as an
ordinary sequence: which tokens would the node's row attend to, and what would its
position be? The flattened tree must give every node that same view. So the mask
row of a node is "the prefix, plus the nodes on my own root path", and nothing
about where the node happens to sit in the flat order matters.

---

You need, for every node, the set of its ancestors and its depth. Both come from
climbing the parent pointers. Do it for all nodes at once: keep a vector `anc`
that starts as `arange(N)` (every node stands on itself). Each step marks
`reach[i, anc[i]] = True` for the nodes that are still inside the tree, counts one
step for them, and moves them up with `anc = parents[anc]`; a node that reaches
`-1` stays there. The loop ends when every node has left, after (maximum depth +
1) steps. `reach` is the tree block of the mask (it includes the diagonal), and
the depth is the number of steps minus one.

---

Assemble the mask from four blocks: prefix-to-prefix is the usual lower triangle,
prefix-to-tree is all `False`, tree-to-prefix is all `True`, tree-to-tree is
`reach`. Position ids are `arange(P)` followed by `P + depth`.

For `verify_tree`, walk: `current = -1`, `want = target_argmax[0]`; while some
node has `parents == current` and `tokens == want`, step onto it and read
`want = target_argmax[node + 1]`. Return the nodes you stepped on and the last
`want`. The `+ 1` is the alignment: entry 0 belongs to the prefix.
