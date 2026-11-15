Suppose you had a suffix automaton of the sequence: the smallest automaton that
accepts exactly its substrings, built online, one token at a time. Each of its
states stands for a set of substrings that all end at the same set of positions.
The state reached by reading the whole sequence ends at exactly one position, the
last one. Which state stands for the longest suffix that ends at **more than one**
position, and how is it connected to the state of the whole sequence?

---

It is the *suffix link* of the state of the whole sequence. The link of a state
points to the state of the longest suffix of its strings that ends at more
positions than they do. So after every `extend`, the match is
`link[last]`, and `match_length()` is the `length` stored in that state: no
searching at all. For the draft you need the earliest position at which the match
ended. Store it per state: a state created for a new token ends first at the
current position; a state created by splitting another one (a *clone*) holds the
same strings the original held, so it ended first where the original did.

---

Per state keep `next` (a dict), `link`, `length`, and `first_end`. To append token
`t` at position `pos`:

```
cur = new state(length = pos + 1, first_end = pos)
p = last
while p exists and t not in next[p]:  next[p][t] = cur;  p = link[p]
if p does not exist:                  link[cur] = root
else:
    q = next[p][t]
    if length[q] == length[p] + 1:    link[cur] = q
    else:
        clone = new state(length = length[p] + 1, next = copy of next[q],
                          link = link[q], first_end = first_end[q])
        while p exists and next[p][t] == q:  next[p][t] = clone;  p = link[p]
        link[q] = link[cur] = clone
last = cur
```

Then `draft` is `tokens[first_end[link[last]] + 1 :][:max_len]`, provided
`length[link[last]] >= min_match`.
