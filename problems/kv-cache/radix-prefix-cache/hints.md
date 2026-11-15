`match_prefix` and `insert` do the same thing for most of their work. What is the
state of the tree after "follow the tokens as far as they go, and if you stop in
the middle of an edge, cut the edge there"? Write that once, and ask what each of
the two methods still has to do afterwards.

---

Give each node its parent, its edge (a list of tokens), a dict of children keyed
by the **first token** of each child's edge, a lock count, and a time. The walk:
look up the child by the next token; count how many tokens of its edge agree with
the input; if that is the whole edge, descend and continue; if not, split and stop.

Split by creating the *upper* half as a new node and letting the existing node
become the lower half. Then an old handle to the existing node still denotes the
same prefix, and its children need no update. The upper node copies the lock count
(whoever locked the lower half or anything below it also depends on the tokens
above) and the time.

For `evict`, collect the unlocked leaves into a min-heap by time; when a removal
leaves its parent without children, push the parent if it is unlocked and not the
root.

---

```
walk(tokens):
    clock += 1; node = root; matched = 0
    while matched < len(tokens):
        child = node.children.get(tokens[matched])
        if child is None: break
        common = length of the common prefix of child.key and tokens[matched:]
        if common < len(child.key):
            child = split(child, common)       # returns the new upper node
        child.time = clock                     # only nodes matched in full
        node = child; matched += common
        if it was split: break
    return matched, node

split(child, c):
    upper = Node(parent=child.parent, key=child.key[:c], time=child.time)
    upper.locks = child.locks
    child.parent.children[upper.key[0]] = upper        # replaces child there
    child.key = child.key[c:]; child.parent = upper
    upper.children[child.key[0]] = child
    return upper

insert(tokens):
    matched, node = walk(tokens)
    if matched < len(tokens): add a leaf under node with key tokens[matched:], time clock
    return matched

lock(node):   while node is not root: node.locks += 1; node = node.parent
```

Stamp the time **after** deciding how much of the edge matched: the lower half of
a split was not used and must keep its old time.
