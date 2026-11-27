The score matrix for one step is rectangular rather than square. It has one row per **new** query and one column per entry in the cache after appending, giving shape `(T, S + T)`. Try `S = 3` and `T = 2`, and write down which of the five columns each of the two rows is allowed to see. What is the absolute sequence position of new token `i`?

---

Concatenate first along axis 2 with cached entries before new ones (skipping concatenation when `cache` is `None`). Column index `j` of the score matrix now equals the entry's absolute position, and new token `i` sits at absolute position `S + i`. The causal rule "a token sees itself and everything before it" becomes `j <= S + i`. When `S = 0` this gives the usual lower triangle, and when `T = 1` every column is visible.

---

Construct the `(T, S + T)` boolean mask by broadcasting two index ranges against each other:

```
visible[i, j] = (j <= S + i)        # arange(S + T)[None, :] <= S + arange(T)[:, None]
```

Pass this mask to `ops.attention(q_new, K, V, visible)`, where `True` means "may attend" and the mask broadcasts over the batch and head axes. Return the attention output along with the concatenated `(K, V)` tuple for the next call.
