Imagine maintaining a suffix automaton of the sequence, built online one token at a time. Each state represents a set of substrings that all end at the same set of positions in the sequence. The state for the entire sequence ends at only one position, the very last index. Which state represents the longest suffix that ends at **more than one** position, and how is it connected to the state of the entire sequence?

---

It is the *suffix link* of the state of the whole sequence. A state's suffix link points to the state of the longest suffix of its strings that ends at more positions than the state itself. After every `extend`, the longest earlier match is therefore `link[last]`, and `match_length()` is simply `length[link[last]]` with no search required. To generate the draft, store `first_end` (the earliest end index) on each state. A newly created state for position `pos` has `first_end = pos`, while a cloned state created during a split inherits `first_end` from the state it was cloned from.

---

For each state, store `next` (a dict), `link`, `length`, and `first_end`. To append token `t` at index `pos`:

```
cur = new state(length = pos + 1, first_end = pos)
p = last
while p exists and t not in next[p]:
    next[p][t] = cur
    p = link[p]
if p does not exist:
    link[cur] = root
else:
    q = next[p][t]
    if length[q] == length[p] + 1:
        link[cur] = q
    else:
        clone = new state(
            length = length[p] + 1,
            next = copy of next[q],
            link = link[q],
            first_end = first_end[q],
        )
        while p exists and next[p][t] == q:
            next[p][t] = clone
            p = link[p]
        link[q] = link[cur] = clone
last = cur
```

Then `draft(max_len, min_match)` returns `tokens[first_end[link[last]] + 1 : first_end[link[last]] + 1 + max_len]` whenever `length[link[last]] >= min_match`, and `[]` otherwise.
