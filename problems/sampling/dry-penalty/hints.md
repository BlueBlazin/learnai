The quantity you need at every position $i$ is "how many tokens ending at $i$ equal
the tokens ending at the end of the window". Reverse the window. What is that
quantity called for a string and its own suffixes, and which classical algorithm
computes it for all positions at once in linear time?

---

In the reversed window `rev`, a run that ended at position $i$ starts at
$k = n - 1 - i$, and the match length is the length of the longest common prefix of
`rev` and `rev[k:]`. The array of those lengths is the **Z-array** of `rev`. The
candidate for position $k$ is the token that followed the earlier run in the
original order, `rev[k - 1]`.

Breakers need no special handling inside the matching: a match is a run at the end
of the window, so count how many tokens at the end of the window are not breakers
and cap every match length at that number.

---

The Z-algorithm keeps the rightmost stretch `[left, right)` of `rev` that is known
to equal a prefix of `rev`. For a new position `k` inside it, the answer at the
mirrored position `k - left` is reused:

```
z = [0] * n; left = right = 0
for k in 1 .. n-1:
    if k < right and z[k - left] < right - k:
        z[k] = z[k - left]                 # the mirrored match ends inside the stretch
    else:
        length = max(0, right - k)         # this much is known to match
        while k + length < n and rev[length] == rev[k + length]: length += 1
        z[k] = length; left, right = k, k + length
```

`right` never moves backward and every comparison that succeeds moves it forward,
so the whole loop is $O(n)$. Then:

```
limit = number of trailing non-breakers of the window
for k in 1 .. n-1:
    m = min(z[k], limit)
    if m >= allowed_length and rev[k - 1] not in breakers:
        longest[rev[k - 1]] = max(longest.get(rev[k - 1], 0), m)
penalty = multiplier * base ** min(m - allowed_length, e_max)
```
