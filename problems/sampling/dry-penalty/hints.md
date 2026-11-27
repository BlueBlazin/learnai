At every position $i$, you need to know how many tokens ending at $i$ match the
tokens ending at the end of the window. If you reverse the window, what is that
quantity called for a string and its suffixes, and which classical algorithm computes
it for all positions in linear time?

---

In the reversed window `rev`, a match that ended at position $i$ starts at
$k = n - 1 - i$. Its match length is the longest common prefix of `rev` and
`rev[k:]`. The array of these lengths is the **Z-array** of `rev`, and the candidate
token that followed the earlier run in original order is `rev[k - 1]`.

Breakers do not need special handling inside the Z-algorithm. Because every match is
a suffix of the window, count how many trailing tokens of the window are not breakers
and cap every match length at that count.

---

The Z-algorithm tracks the rightmost interval `[left, right)` of `rev` that is known
to match a prefix of `rev`. For a new index `k` inside that interval, reuse the
already-computed value at the mirrored index `k - left`:

```
z = [0] * n; left = right = 0
for k in 1 .. n-1:
    if k < right and z[k - left] < right - k:
        z[k] = z[k - left]                 # the mirrored match ends inside the interval
    else:
        length = max(0, right - k)         # this much is already known to match
        while k + length < n and rev[length] == rev[k + length]: length += 1
        z[k] = length; left, right = k, k + length
```

Because `right` never decreases and every successful comparison advances it, the loop
runs in $O(n)$ time. Afterward, apply the breaker limit and compute the penalties:

```
limit = number of trailing non-breakers of the window
for k in 1 .. n-1:
    m = min(z[k], limit)
    if m >= allowed_length and rev[k - 1] not in breakers:
        longest[rev[k - 1]] = max(longest.get(rev[k - 1], 0), m)
penalty = multiplier * base ** min(m - allowed_length, e_max)
```
