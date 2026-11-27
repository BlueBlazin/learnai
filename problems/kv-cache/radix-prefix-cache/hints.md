`match_prefix` and `insert` share almost all of their logic. Suppose you write a helper that follows `tokens` down the tree as far as they match and splits the final edge if the match stops partway through. What does the tree look like right after that helper returns, and what extra step does each method need afterward?

---

Give each node a `parent` pointer, an edge token list, a `children` dictionary keyed by the **first token** of each child's edge, a lock count, and a last access time. During the walk, look up the next child using the current token and count how many tokens of its edge match the input. If the whole edge matches, descend and continue. Otherwise, split the edge and stop.

When splitting, create a new node for the *upper* half and keep the existing node as the lower half. That way, any existing handle pointing to the node still represents the same full prefix, and its children do not need to be reattached. Copy both the lock count and the last access time to the upper node, since any request locking the lower half or a descendant also depends on the tokens above it.

For `evict`, collect all unlocked leaves into a min-heap ordered by last access time. Whenever removing a leaf leaves its parent with no children, push the parent onto the heap if it is unlocked and not the root.

---

```
walk(tokens):
    clock += 1
    node = root
    matched = 0
    while matched < len(tokens):
        child = node.children.get(tokens[matched])
        if child is None:
            break
        common = length of the common prefix of child.key and tokens[matched:]
        if common < len(child.key):
            child = split(child, common)       # returns the new upper node
        child.time = clock                     # only nodes matched in full
        node = child
        matched += common
        if it was split:
            break
    return matched, node

split(child, c):
    upper = Node(parent=child.parent, key=child.key[:c], time=child.time)
    upper.locks = child.locks
    child.parent.children[upper.key[0]] = upper        # replaces child there
    child.key = child.key[c:]
    child.parent = upper
    upper.children[child.key[0]] = child
    return upper

insert(tokens):
    matched, node = walk(tokens)
    if matched < len(tokens):
        add a leaf under node with key tokens[matched:] and time clock
    return matched

lock(node):
    while node is not root:
        node.locks += 1
        node = node.parent
```

Stamp the access time **after** determining how much of the edge matched. The unmatched lower half of a split edge was not used and must keep its previous access time.
